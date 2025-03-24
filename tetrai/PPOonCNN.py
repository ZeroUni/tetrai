import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import numpy as np
import time
import os
import json
import argparse
import traceback
import threading
import socket
import queue
from collections import deque
from typing import Dict, List, Optional, Tuple, Any, Union

import TetrisCNN
import TetrisEnv
from rewardNormalizer import RewardNormalizer
from trainingUtils import DisplayManager
from streamingServer import TetrisStreamServer

# Add import for TrainingDashboard
from streamingServer import TrainingDashboard

class PPOBuffer:
    def __init__(self, capacity=2048, gamma=0.99, lam=0.95, device='cuda'):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.capacity = capacity
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.device = device
        
    def push(self, state, action, reward, value, log_prob, done):
        # Expand buffer if needed
        if len(self.states) < self.capacity:
            self.states.append(None)
            self.actions.append(None)
            self.rewards.append(None)
            self.values.append(None)
            self.log_probs.append(None)
            self.dones.append(None)
        
        # Store transition
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        
    def compute_advantages(self, last_value=0.0):
        size = len(self.rewards)
        advantages = torch.zeros(size, device=self.device)
        last_gae_lam = 0
        
        with torch.cuda.stream(torch.cuda.Stream()):
            # Convert to tensors for efficiency
            rewards = torch.tensor(self.rewards, device=self.device)
            values = torch.tensor(self.values + [last_value], device=self.device)
            dones = torch.tensor(self.dones, device=self.device, dtype=torch.float32)
            
            # Compute GAE advantages in reverse
            for t in reversed(range(size)):
                if t == size - 1:
                    next_non_terminal = 1.0 - dones[t]
                    next_value = last_value
                else:
                    next_non_terminal = 1.0 - dones[t]
                    next_value = values[t + 1]
                
                delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
                last_gae_lam = delta + self.gamma * self.lam * next_non_terminal * last_gae_lam
                advantages[t] = last_gae_lam
                
            # Compute returns
            returns = advantages + torch.tensor(self.values, device=self.device)
            normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            return advantages, returns, normalized_advantages
            
    def get_batch(self):
        # Convert lists to tensors
        states = torch.stack(self.states)
        actions = torch.tensor(self.actions, device=self.device)
        log_probs = torch.tensor(self.log_probs, device=self.device)
        
        advantages, returns, normalized_advantages = self.compute_advantages()
        
        return states, actions, log_probs, returns, normalized_advantages
        
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.ptr = 0
        torch.cuda.empty_cache()

def preprocess_state_batch(states, frame_stack=None, preprocess_pool=None):
    """
    Optimized state preprocessing using pure tensor operations with optional pooling
    
    Args:
        states: Input tensor of shape [B, H, W] or [B, 1, H, W]
        frame_stack: Optional FrameStack object
        preprocess_pool: Optional PreprocessingPool for memory efficiency
        
    Returns:
        Processed tensor of shape [B, C, 128, 128]
    """
    # Ensure proper input shape and type
    if states.dim() == 3:
        states = states.unsqueeze(1)  # Add channel dimension
    if states.dim() == 2:
        states = states.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
    # Add batch dimension if needed
    if states.dim() == 3:
        states = states.unsqueeze(0)
        
    # Normalize and convert to float16 in one operation
    states = states.to(dtype=torch.float16, device='cuda') / 255.0
    
    # Try to get tensor from pool
    output = None
    if preprocess_pool is not None:
        output = preprocess_pool.get_tensor()
    
    if output is None:
        # Fallback to direct processing
        # Resize first
        states = F.interpolate(states, 
                           size=(128, 128),
                           mode='nearest')
        # Convert memory format
        states = states.to(memory_format=torch.channels_last)
    else:
        # Use pooled tensor
        with torch.cuda.stream(torch.cuda.current_stream()):
            # Resize first
            states = F.interpolate(states,
                               size=(128, 128),
                               mode='nearest')
            # Ensure states has correct shape [B, C, H, W]
            if states.dim() < 4:
                states = states.unsqueeze(0)
            # Reshape output if needed to match states
            if output.shape != states.shape:
                output = output.view_as(states)
            # Copy to output tensor with proper memory format
            output.copy_(states)
            states = output.to(memory_format=torch.channels_last)
    
    if frame_stack is not None:
        states = frame_stack(states)
        states = states.reshape(-1, 4, 128, 128)
    
    # Return tensor to pool if we used one
    if output is not None:
        preprocess_pool.return_tensor(output)
        
    return states

class GameBuffer:
    def __init__(self, ppo_buffer, gamma=0.99, lookback=10):
        self.ppo_buffer = ppo_buffer
        self.lookback = lookback
        self.recent_states = []
        self.recent_actions = []
        self.recent_values = []
        self.recent_log_probs = []
        self.recent_next_states = []
        self.recent_dones = []
        
    def add_transition(self, state, action, value, log_prob, next_state, done):
        self.recent_states.append(state)
        self.recent_actions.append(action)
        self.recent_values.append(value)
        self.recent_log_probs.append(log_prob)
        self.recent_next_states.append(next_state)
        self.recent_dones.append(done)
        
        # Process oldest transition if exceeding lookback
        if len(self.recent_states) > self.lookback:
            self.process_excess_transitions()

    def process_excess_transitions(self, count=1):
        """Process oldest transitions by adding them to the PPO buffer"""
        for _ in range(min(count, len(self.recent_states) - 1)):
            self.ppo_buffer.push(
                self.recent_states.pop(0),
                self.recent_actions.pop(0),
                0.0,  # No immediate reward since we're still evaluating
                self.recent_values.pop(0),
                self.recent_log_probs.pop(0),
                self.recent_dones.pop(0)
            )
            self.recent_next_states.pop(0)
            
    def process_game_over(self, final_reward, lines_cleared):
        """Process game over by distributing negative reward across recent moves"""
        n = len(self.recent_states)
        if n == 0:
            return
            
        # Add lines_cleared bonus to reward, with a max reward of 0
        final_reward = min(10, final_reward + 100 * lines_cleared)
        reward_per_move = final_reward / n
        
        # Process all remaining transitions with distributed reward
        while len(self.recent_states) > 0:
            temporal_factor = len(self.recent_states) / n  # Earlier moves get smaller factor
            move_reward = reward_per_move * temporal_factor
            
            self.ppo_buffer.push(
                self.recent_states.pop(0),
                self.recent_actions.pop(0),
                move_reward,
                self.recent_values.pop(0),
                self.recent_log_probs.pop(0),
                self.recent_dones.pop(0)
            )
            if self.recent_next_states:
                self.recent_next_states.pop(0)
            
    def process_reward(self, reward):
        """Process a positive reward by distributing it across recent moves without removing states"""
        n = len(self.recent_states)
        if n == 0:
            return
            
        reward_per_move = reward / n
        
        # Process all transitions with distributed reward
        for i in range(n):
            temporal_factor = (n - i) / n  # Later moves get larger factor
            move_reward = reward_per_move * temporal_factor
            
            self.ppo_buffer.push(
                self.recent_states[i],
                self.recent_actions[i],
                move_reward,
                self.recent_values[i],
                self.recent_log_probs[i],
                self.recent_dones[i]
            )
            
        # Don't clear the buffer - states will be removed naturally through lookback or explicit clear
            
    def clear(self):
        self.recent_states.clear()
        self.recent_actions.clear()
        self.recent_values.clear()
        self.recent_log_probs.clear()
        self.recent_next_states.clear()
        self.recent_dones.clear()

class TaskQueue:
    """Thread-safe task queue for dynamic episode allocation"""
    def __init__(self, total_episodes: int, min_batch_size: int = 1):
        self.queue = queue.Queue()
        self.remaining = total_episodes
        self.completed = 0
        self.in_progress = 0
        self.min_batch_size = min_batch_size
        self.lock = threading.Lock()
        self.results = []
        self.total_episodes = total_episodes  # Store the total episodes
    
    def initialize(self):
        """Reset the queue for a new training iteration"""
        with self.lock:
            self.queue = queue.Queue()
            self.in_progress = 0
            self.completed = 0
            # Don't reset total_episodes here
    
    def get_task(self) -> Optional[int]:
        """Get the next task (number of episodes to process)
        
        Returns:
            Number of episodes to process, or None if no more work
        """
        with self.lock:
            if self.remaining <= 0:
                return None
                
            # Dynamically determine batch size based on remaining episodes
            # Start with smaller batches and increase when we have fewer workers
            if self.in_progress > 0:
                # If other workers are busy, take a smaller batch
                batch_size = max(self.min_batch_size, min(3, self.remaining))
            else:
                # If we're the only/first worker, take a larger batch
                batch_size = max(self.min_batch_size, min(5, self.remaining))
            
            # Ensure we don't exceed remaining episodes
            batch_size = min(batch_size, self.remaining)
            
            if batch_size > 0:
                self.remaining -= batch_size
                self.in_progress += batch_size
                return batch_size
            return None
    
    def submit_result(self, num_episodes: int, rewards: List[float], steps: List[int]):
        """Submit results from completed episodes"""
        with self.lock:
            self.completed += num_episodes
            self.in_progress -= num_episodes
            self.results.append((rewards, steps))
    
    def is_complete(self) -> bool:
        """Check if all episodes have been completed"""
        with self.lock:
            return self.completed == self.total_episodes and self.in_progress == 0
    
    def get_results(self) -> Tuple[List[float], List[int]]:
        """Get all collected results
        
        Returns:
            Tuple of (rewards, steps)
        """
        all_rewards = []
        all_steps = []
        
        for rewards, steps in self.results:
            all_rewards.extend(rewards)
            all_steps.extend(steps)
            
        return all_rewards, all_steps
    
    def get_progress(self) -> Tuple[int, int, int]:
        """Get current progress
        
        Returns:
            Tuple of (completed, in_progress, remaining)
        """
        with self.lock:
            return self.completed, self.in_progress, self.remaining

class Worker:
    """Worker for parallel data collection"""
    def __init__(self, 
                 worker_id: int, 
                 shared_buffer: 'PPOBuffer', 
                 model: torch.nn.Module,
                 device: torch.device,
                 max_moves: int, 
                 weights: Optional[Dict] = None,
                 level: int = 1,
                 display_manager: Optional[DisplayManager] = None,
                 record: bool = False,
                 display_enabled: bool = True,  # Add this parameter
                 stream_port: Optional[int] = None,
                 debug: bool = False):
        self.worker_id = worker_id
        self.shared_buffer = shared_buffer
        self.model = model
        self.device = device
        self.max_moves = max_moves
        self.debug = debug
        self.record = record
        self.display_enabled = display_enabled  # Store display_enabled flag
        
        # Create separate display manager for recording without visual display if display_enabled=False
        if self.record and display_manager is None:
            self.recording_manager = DisplayManager(
                record_video=True,
                video_filename=f"worker_{worker_id}",
                headless=not display_enabled  # Always use headless mode when display_enabled is False
            )
            self.recording_manager.start()
        else:
            self.recording_manager = None
            
        # Use provided display manager if given
        self.display_manager = display_manager or self.recording_manager
        
        # Create worker-specific environment
        self.env = TetrisEnv.TetrisEnv(
            display_manager=self.display_manager, 
            max_moves=max_moves, 
            weights=weights, 
            level=level
        )
        # Only enable rendering if display is enabled and we have a display manager
        self.env.render_mode = self.display_manager is not None and display_enabled
        
        # Create worker-specific buffers and helpers
        self.game_buffer = GameBuffer(shared_buffer, gamma=0.99, lookback=15)
        self.reward_normalizer = RewardNormalizer()
        
        # Create worker-specific frame stack and preprocessing pool
        from DDQNonCNN import FrameStack, PreprocessingPool
        self.frame_stack = FrameStack(stack_size=4)
        self.preprocess_pool = PreprocessingPool(size=32)
        
        # Thread control
        self.thread = None
        self.running = False
        self.result_queue = queue.Queue()
        
        # Setup streaming if requested
        self.stream_server = None
        if stream_port is not None:
            self.stream_server = TetrisStreamServer(port=stream_port, worker_id=worker_id)
            self.stream_server.start()

        # Statistics tracking
        self.stats = {
            'episodes_completed': 0,
            'total_reward': 0,
            'rewards': [],
            'episode_lengths': [],
            'mean_reward': 0
        }
        
    def run_episode(self) -> Tuple[float, int]:
        """Run a single episode and collect data"""
        self.model.eval()  # Set to eval mode for inference
        
        # Reset environment and frame stack
        state = self.env.reset()
        self.frame_stack.reset()
        state = preprocess_state_batch(state, self.frame_stack, self.preprocess_pool).to(self.device)
        
        episode_reward = 0
        done = False
        episode_steps = 0
        
        # Collect trajectory
        while not done and episode_steps < self.max_moves:
            # Sample action from policy
            with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
                state_input = state.reshape(-1, 4, 128, 128)
                action_probs, log_probs, state_value = self.model.ppo_forward(state_input)
                
                # Apply action masking
                action_mask = self.env.get_legal_actions_mask().to(self.device)
                masked_probs = action_probs * action_mask
                
                # Safety check to prevent NaN values
                mask_sum = masked_probs.sum(dim=-1, keepdim=True)
                if mask_sum.item() > 0:
                    masked_probs = masked_probs / mask_sum
                else:
                    # Fallback: use uniform distribution over legal actions
                    masked_probs = action_mask / (action_mask.sum(dim=-1, keepdim=True) + 1e-8)
                    
                    # Log the issue
                    if self.debug:
                        print(f"Worker {self.worker_id}: All masked probabilities were zero")
                
                # Sample action from distribution
                dist = torch.distributions.Categorical(masked_probs)
                action = dist.sample().item()
                
                # Get log probability of the action
                action_log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()
            
            # Execute action in environment
            next_state, reward, done = self.env.step(action)
            episode_steps += 1
            
            # Process state
            next_state = preprocess_state_batch(next_state, self.frame_stack, self.preprocess_pool)
            next_state = next_state.to(self.device)
            
            # Update reward
            episode_reward += reward
            
            # Store transition in game buffer
            self.game_buffer.add_transition(
                state.clone(),
                action,
                state_value.item(),
                action_log_prob,
                next_state,
                done
            )
            
            # If we got a reward or the episode is done, process the game buffer
            if reward != 0 or done:
                if self.env.game_state.game_over:
                    # Process game over sequence
                    self.game_buffer.process_game_over(reward - 100, self.env.game_state.lines_cleared)
                else:
                    # Process regular reward sequence
                    self.game_buffer.process_reward(reward)
            
            # Send current frame to stream server if active
            if self.stream_server and episode_steps % 2 == 0:  # Every other frame to reduce load
                self.stream_server.update_frame(self.env.get_state().cpu().numpy())
            
            state = next_state
        
        # Reset reward normalizer
        self.reward_normalizer.reset()
        
        # Update stats
        self.stats['episodes_completed'] += 1
        self.stats['total_reward'] += episode_reward
        self.stats['rewards'].append(episode_reward)
        self.stats['episode_lengths'].append(episode_steps)
        self.stats['mean_reward'] = self.stats['total_reward'] / self.stats['episodes_completed']
        
        return episode_reward, episode_steps
        
    def run_episodes_thread(self, task_queue: 'TaskQueue'):
        """Run episodes from a shared task queue"""
        # Create local references to frame_stack and preprocess_pool to prevent race conditions
        frame_stack = self.frame_stack
        preprocess_pool = self.preprocess_pool
        
        while True:
            # Try to get a task from the queue
            num_episodes = task_queue.get_task()
            if num_episodes is None:
                break
                
            rewards = []
            steps = []
            
            for _ in range(num_episodes):
                if not self.running:
                    # Submit any completed episodes
                    if rewards:
                        task_queue.submit_result(len(rewards), rewards, steps)
                    return
                    
                try:
                    # Use the local references instead of the instance variables
                    if not hasattr(self, 'frame_stack') or self.frame_stack is None:
                        print(f"Worker {self.worker_id}: Recreating frame stack")
                        from DDQNonCNN import FrameStack, PreprocessingPool
                        self.frame_stack = FrameStack(stack_size=4)
                        self.preprocess_pool = PreprocessingPool(size=32)
                        frame_stack = self.frame_stack
                        preprocess_pool = self.preprocess_pool
                    
                    reward, step_count = self.run_episode()
                    rewards.append(reward)
                    steps.append(step_count)
                    
                    # Clear CUDA cache periodically
                    if len(rewards) % 5 == 0:
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Worker {self.worker_id} error: {e}")
                    traceback.print_exc()
                    break
            
            # Submit results for the completed batch
            if rewards:
                task_queue.submit_result(len(rewards), rewards, steps)
    
    def start_with_queue(self, task_queue: 'TaskQueue'):
        """Start worker with a shared task queue"""
        if self.thread is not None and self.thread.is_alive():
            return
            
        self.running = True
        self.thread = threading.Thread(
            target=self.run_episodes_thread,
            args=(task_queue,),
            daemon=True
        )
        self.thread.start()
        
    def stop(self):
        """Stop worker thread"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=5.0)
            self.thread = None
    
    def get_results(self, timeout: float = 0.1) -> Optional[Tuple[List[float], List[int]]]:
        """Get results from the worker, returns None if no results yet"""
        try:
            return self.result_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None
    
    def is_alive(self) -> bool:
        """Check if worker thread is still running"""
        return self.thread is not None and self.thread.is_alive()
    
    def increase_level(self):
        """Increase game level"""
        self.env.increase_level()
    
    def close(self):
        """Clean up resources"""
        self.stop()
        
        if self.env:
            self.env.close()
            
        if self.display_manager and self.display_manager == self.recording_manager:
            self.display_manager.stop()
            
        if self.stream_server:
            self.stream_server.stop()
            
        # Use a more careful approach to clearing memory
        fs = self.frame_stack
        pp = self.preprocess_pool
        gb = self.game_buffer
        
        # Set to None first, then delete
        self.frame_stack = None
        self.preprocess_pool = None
        self.game_buffer = None
        
        # Now delete the local references
        try:
            del fs
            del pp
            del gb
        except Exception as e:
            print(f"Error cleaning up Worker {self.worker_id} resources: {e}")
        
        torch.cuda.empty_cache()

    def get_state_dict(self) -> Dict:
        """Get worker state for checkpointing"""
        return {
            'worker_id': self.worker_id,
            'stats': self.stats,
            'level': self.env.get_level()
        }
        
    def load_state_dict(self, state_dict: Dict):
        """Load worker state from checkpoint"""
        if 'stats' in state_dict:
            self.stats = state_dict['stats']
        if 'level' in state_dict:
            # Reset env with the right level
            if self.env.get_level() != state_dict['level']:
                self.env = TetrisEnv.TetrisEnv(
                    display_manager=self.display_manager, 
                    max_moves=self.max_moves, 
                    level=state_dict['level']
                )
                self.env.render_mode = self.display_manager is not None

# Fix the worker initialization in main function
def main(
    num_episodes=1000,
    batch_size=64,
    gamma=0.99,
    lam=0.95,
    clip_ratio=0.2,
    target_kl=0.01,
    value_coef=0.5,
    entropy_coef=0.1,
    max_moves=100,
    save_interval=100,
    weights=None,
    learning_rate=3e-4,
    display_enabled=True,
    record=False,
    level=1,
    level_inc=-1,
    cycles=1,
    debug=False,
    num_workers=1,
    display_worker=0,
    checkpoint=None,
    stream_base_port=None
):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # Create unique output directory
    timestamp = int(time.time())
    model_dir = f'out/{timestamp}'
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize display manager if needed (only for display worker)
    display_manager = None
    if display_enabled:
        display_manager = DisplayManager(record_video=record and display_worker == 0, headless=False)
        display_manager.start()
    
    # Load weights for environment
    if weights:
        if isinstance(weights, str):  # Load from file
            with open(weights, 'r') as f:
                weights = json.load(f)
        elif isinstance(weights, dict):  # Use directly
            pass
    
    # Create model
    model = TetrisCNN.RainbowTetrisCNN(
        num_actions=8,
        device=device,
        num_atoms=51,
        v_min=-10,
        v_max=10
    ).to(device, memory_format=torch.channels_last)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)
    scaler = GradScaler()
    
    # Set up shared PPO buffer
    ppo_buffer = PPOBuffer(capacity=2048 * num_workers, gamma=gamma, lam=lam, device=device)
    
    # Setup scheduler
    temp_reset = num_episodes // cycles
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=temp_reset,
        T_mult=1,
        eta_min=learning_rate * 0.1
    )

    # Initial parameters
    base_entropy_coef = entropy_coef
    min_entropy_coef = 0.1 * base_entropy_coef
    base_clip_ratio = clip_ratio
    min_clip_ratio = 0.05
    
    # Initialize training dashboard if streaming is enabled
    training_dashboard = None
    if stream_base_port is not None:
        training_dashboard = TrainingDashboard(base_port=stream_base_port)
        training_dashboard.start()
        
        # Set initial stats with complete training parameters
        training_dashboard.update_stats({
            'episodes_completed': 0,
            'total_episodes': num_episodes,
            'avg_reward': 0.0,
            'avg_steps': 0.0,
            'current_cycle': 1,
            'total_cycles': cycles,
            'status': 'Running',
            'reward_x': [0],
            'reward_y': [0]
        })
    
    # Initialize workers
    workers = []
    for i in range(num_workers):
        worker_display = display_manager if i == display_worker and display_enabled else None
        
        # Calculate worker stream port: base_port + worker_id + 1
        # This reserves base_port for the dashboard
        worker_stream_port = None
        if stream_base_port is not None:
            worker_stream_port = stream_base_port + i + 1
            if training_dashboard:
                training_dashboard.add_worker(i, worker_stream_port)
        
        # Determine if this worker should record
        should_record = record and (not display_enabled or i != display_worker)
        
        worker = Worker(
            worker_id=i,
            shared_buffer=ppo_buffer,
            model=model,
            device=device,
            max_moves=max_moves,
            weights=weights,
            level=level,
            display_manager=worker_display,
            record=should_record,  # Record based on display_enabled status
            display_enabled=display_enabled,  # Pass the display_enabled flag
            stream_port=worker_stream_port,
            debug=debug
        )
        workers.append(worker)
        print(f"Initialized worker {i}, display: {worker_display is not None and display_enabled}, record: {should_record}, stream port: {worker_stream_port}")
    
    # Initialize task queue
    task_queue = TaskQueue(total_episodes=num_episodes, min_batch_size=1)
    
    # Results tracking
    results = {
        'episode_rewards': [],
        'policy_losses': [],
        'value_losses': [],
        'entropy_losses': [],
        'learning_rates': [],
        'episode_steps': [],
        'kl_divs': [],
        'clip_ratios': [],
        'entropy_coefs': [],
        'workers': [worker.get_state_dict() for worker in workers]
    }
    
    # Load checkpoint if provided
    episodes_completed = 0
    if checkpoint:
        try:
            print(f"Loading checkpoint: {checkpoint}")
            checkpoint_data = torch.load(checkpoint)
            
            # Determine the checkpoint format (new or legacy format)
            if isinstance(checkpoint_data, dict):
                # New format: {episodes_completed, model_state_dict, optimizer_state_dict, ...}
                if 'model_state_dict' in checkpoint_data:
                    model.load_state_dict(checkpoint_data['model_state_dict'])
                    print("Loaded model state from checkpoint")
                else:
                    print("Warning: model_state_dict not found in checkpoint, trying direct load")
                    # Legacy format or direct model state dict
                    model.load_state_dict(checkpoint_data)
                    
                # Load optimizer state if available
                if 'optimizer_state_dict' in checkpoint_data:
                    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                    print("Loaded optimizer state from checkpoint")
                
                # Load scheduler state if available
                if 'scheduler_state_dict' in checkpoint_data:
                    scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                    print("Loaded scheduler state from checkpoint")
                
                # Load episode count if available
                if 'episodes_completed' in checkpoint_data:
                    episodes_completed = checkpoint_data['episodes_completed']
                    print(f"Loaded episode count: {episodes_completed}")
                
                # Load results if available
                if 'results' in checkpoint_data:
                    results = checkpoint_data['results']
                    print("Loaded training results from checkpoint")
                    
                    # Load worker states if available
                    if 'workers' in results:
                        worker_states = results['workers']
                        for i, worker in enumerate(workers):
                            if i < len(worker_states):
                                worker.load_state_dict(worker_states[i])
                                print(f"Loaded state for worker {i}")
            else:
                # Directly use checkpoint data as model state dict
                model.load_state_dict(checkpoint_data)
                print("Loaded model state (legacy format)")
                
            print(f"Successfully loaded checkpoint at episode {episodes_completed}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            traceback.print_exc()
            print("Proceeding with fresh model")
    
    # Training loop
    try:
        best_reward = float('-inf')
        last_checkpoint_episode = 0  # Track last checkpoint episode
        last_checkpoint_time = time.time()  # Track last checkpoint time
        
        while episodes_completed < num_episodes:
            # Calculate entropy and clip ratio based on cycle progress
            cycle_idx = episodes_completed // temp_reset
            cycle_position = episodes_completed % temp_reset
            cycle_progress = cycle_position / temp_reset

            current_entropy_coef = min_entropy_coef + (base_entropy_coef - min_entropy_coef) * (0.5 * (1 + np.cos(np.pi * cycle_progress)))
            current_clip_ratio = min_clip_ratio + (base_clip_ratio - min_clip_ratio) * (0.5 * (1 + np.cos(np.pi * cycle_progress)))
            
            # Reset buffer at the start of each cycle
            if episodes_completed > 0 and episodes_completed % temp_reset == 0:
                print(f"Starting new cycle {episodes_completed // temp_reset}")
                ppo_buffer.clear()
                print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
            
            # Increase level if needed
            if episodes_completed > 0 and level_inc != -1 and episodes_completed % level_inc == 0:
                for worker in workers:
                    worker.increase_level()
                print(f"Increased level at episode {episodes_completed}")
            
            # Initialize task queue for this iteration
            remaining_episodes = num_episodes - episodes_completed
            task_queue.total_episodes = remaining_episodes  # Ensure total_episodes is set correctly
            task_queue.remaining = remaining_episodes
            task_queue.results = []
            task_queue.initialize()
            
            # Start all workers with the shared queue
            for worker in workers:
                worker.start_with_queue(task_queue)
            
            # Monitor progress and display updates
            last_progress_time = time.time()
            last_dashboard_update = time.time()
            
            while not task_queue.is_complete():
                time.sleep(0.1)  # Short sleep to prevent CPU spinning
                
                # Check if any workers crashed
                all_alive = False
                for worker in workers:
                    if worker.is_alive():
                        all_alive = True
                        break
                
                if not all_alive:
                    print("All workers stopped unexpectedly")
                    break
                
                # Print progress update periodically
                current_time = time.time()
                if current_time - last_progress_time > 5.0:  # Update every 5 seconds
                    completed, in_progress, remaining = task_queue.get_progress()
                    print(f"Progress: {completed} completed, {in_progress} in progress, {remaining} remaining")
                    last_progress_time = current_time
                    
                    # Update dashboard periodically with status even if no episodes are complete
                    if training_dashboard and current_time - last_dashboard_update > 10.0:
                        training_dashboard.update_stats({
                            'episodes_completed': episodes_completed + completed,
                            'status': 'Running',
                            'in_progress': in_progress
                        })
                        last_dashboard_update = current_time
            
            # Get collected results
            all_rewards, all_steps = task_queue.get_results()
            
            # Update episode counter
            episodes_completed += len(all_rewards)
            
            # Add to results
            results['episode_rewards'].extend(all_rewards)
            results['episode_steps'].extend(all_steps)
            results['workers'] = [worker.get_state_dict() for worker in workers]
            
            # Update policy if we have enough data
            if len(ppo_buffer.states) >= ppo_buffer.capacity // 2:
                model.train()
                
                # Compute last value if episode isn't done
                with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
                    if len(ppo_buffer.states) > 0:
                        last_state = ppo_buffer.states[-1].unsqueeze(0)
                        _, _, last_value = model.ppo_forward(last_state)
                        last_value = last_value.item()
                    else:
                        last_value = 0.0
                
                # Get batch data
                states, actions, old_log_probs, returns, advantages = ppo_buffer.get_batch()
                
                # PPO update (multiple epochs)
                policy_losses = []
                value_losses = []
                entropy_losses = []
                kl_divs = []
                
                for _ in range(4):  # Number of PPO epochs
                    # Process in minibatches
                    indices = torch.randperm(len(states))
                    for start_idx in range(0, len(states), batch_size):
                        # Get minibatch
                        idx = indices[start_idx:start_idx + batch_size]
                        mb_states = states[idx]
                        mb_actions = actions[idx]
                        mb_old_log_probs = old_log_probs[idx]
                        mb_returns = returns[idx]
                        mb_advantages = advantages[idx]
                        
                        with autocast(device_type='cuda', dtype=torch.float16):
                            # Forward pass
                            new_action_probs, new_log_probs, values = model.ppo_forward(mb_states)
                            
                            # Get new log probs for taken actions
                            dist = torch.distributions.Categorical(new_action_probs)
                            new_log_probs_actions = dist.log_prob(mb_actions)
                            entropy = dist.entropy().mean()
                            
                            # Calculate ratio and clipped objective
                            ratio = torch.exp(new_log_probs_actions - mb_old_log_probs)
                            surr1 = ratio * mb_advantages
                            surr2 = torch.clamp(ratio, 1.0 - current_clip_ratio, 1.0 + current_clip_ratio) * mb_advantages
                            policy_loss = -torch.min(surr1, surr2).mean()
                            
                            # Value loss
                            value_loss = F.mse_loss(values, mb_returns)
                            
                            # Total loss
                            loss = policy_loss + value_coef * value_loss - current_entropy_coef * entropy
                            
                            # Compute approximate KL divergence for early stopping
                            with torch.no_grad():
                                log_ratio = new_log_probs_actions - mb_old_log_probs
                                kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                                kl_divs.append(kl)
                        
                        # Optimization step with mixed precision
                        optimizer.zero_grad()
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                        scaler.step(optimizer)
                        scaler.update()
                        
                        policy_losses.append(policy_loss.item())
                        value_losses.append(value_loss.item())
                        entropy_losses.append(entropy.item())
                        
                        # Early stopping based on KL divergence
                        if kl > target_kl:
                            break
                            
                    # Early stopping based on KL divergence (epoch level)
                    if kl > target_kl:
                        break
                
                # Clear buffer after update
                ppo_buffer.clear()
                
                # Update results
                results['policy_losses'].append(np.mean(policy_losses))
                results['value_losses'].append(np.mean(value_losses))
                results['entropy_losses'].append(np.mean(entropy_losses))
                results['learning_rates'].append(optimizer.param_groups[0]['lr'])
                results['kl_divs'].append(np.mean(kl_divs))
                results['clip_ratios'].append(current_clip_ratio)
                results['entropy_coefs'].append(current_entropy_coef)
            
            # Update learning rate
            scheduler.step()
            
            # Improved checkpoint logic
            current_time = time.time()
            should_checkpoint = False
            
            # Check if we've passed a save_interval threshold
            if episodes_completed >= last_checkpoint_episode + save_interval:
                should_checkpoint = True
                
            # Also ensure we save at least every 30 minutes regardless of episodes
            if current_time - last_checkpoint_time > 1800:  # 30 minutes
                should_checkpoint = True
                
            if should_checkpoint:
                checkpoint_path = f'{model_dir}/tetris_ppo_{episodes_completed}.pth'
                
                # Wait for all workers to be idle before saving the checkpoint
                if task_queue.in_progress > 0:
                    print(f"Waiting for {task_queue.in_progress} tasks to complete before checkpointing...")
                    while task_queue.in_progress > 0 and time.time() - current_time < 30:
                        time.sleep(0.5)  # Wait up to 30 seconds for tasks to complete
                
                # Now safe to checkpoint
                torch.save({
                    'episodes_completed': episodes_completed,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'results': results
                }, checkpoint_path)
                
                # Also save results as JSON for easier analysis
                with open(f'{model_dir}/training_results.json', 'w') as f:
                    json.dump(results, f)
                
                print(f"Saved checkpoint at episode {episodes_completed} to {checkpoint_path}")
                last_checkpoint_episode = episodes_completed
                last_checkpoint_time = time.time()
            
            # Print progress
            avg_reward = np.mean(all_rewards) if all_rewards else 0
            avg_steps = np.mean(all_steps) if all_steps else 0
            print(f"Completed {episodes_completed}/{num_episodes} episodes: Avg Reward = {avg_reward:.2f}, Avg Steps = {avg_steps:.2f}, Buffer Size = {len(ppo_buffer.states)}")
            
            # Check for improvement (early stopping)
            if avg_reward > best_reward and len(all_rewards) > 0:
                best_reward = avg_reward
                torch.save({
                    'episodes_completed': episodes_completed,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'results': results
                }, f'{model_dir}/tetris_ppo_best.pth')
            
            # Update dashboard with latest stats
            if training_dashboard:
                # Calculate window stats
                window_size = min(10, len(results['episode_rewards']))
                recent_rewards = results['episode_rewards'][-window_size:] if results['episode_rewards'] else [0]
                recent_steps = results['episode_steps'][-window_size:] if results['episode_steps'] else [0]
                
                # Prepare data for charts
                reward_x = list(range(len(results['episode_rewards'])))
                reward_y = results['episode_rewards']
                
                # Use rolling window for smoother chart
                if len(reward_y) > 10:
                    smoothed_y = []
                    window = 10
                    for i in range(window-1, len(reward_y)):
                        smoothed_y.append(sum(reward_y[i-window+1:i+1])/window)
                    reward_y = smoothed_y
                    reward_x = reward_x[window-1:]
                
                # Get metrics data
                metrics_x = list(range(len(results['policy_losses'])))
                
                # Send stats update
                training_dashboard.update_stats({
                    'episodes_completed': episodes_completed,
                    'total_episodes': num_episodes,
                    'avg_reward': sum(recent_rewards) / len(recent_rewards),
                    'avg_steps': sum(recent_steps) / len(recent_steps),
                    'current_cycle': episodes_completed // temp_reset + 1,
                    'total_cycles': cycles,
                    'status': 'Running',
                    'reward_x': reward_x,
                    'reward_y': reward_y,
                    'metrics_x': metrics_x,
                    'metrics_y': {
                        'policy': results['policy_losses'],
                        'value': results['value_losses'],
                        'entropy': results['entropy_losses']
                    }
                })
            
            # Explicit cleanup
            torch.cuda.empty_cache()
        
        # Final save
        torch.save({
            'episodes_completed': episodes_completed,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'results': results
        }, f'{model_dir}/tetris_ppo_final.pth')
        
        with open(f'{model_dir}/training_results.json', 'w') as f:
            json.dump(results, f)
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Cleaning up...")
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
    finally:
        print("Cleaning up resources...")
        # Stop all workers first - improved cleanup sequence
        for worker in workers:
            try:
                # First, clear any resources that depend on GLFW
                if worker.display_manager:
                    worker.display_manager.running = False
                worker.stop()
            except Exception as e:
                print(f"Error stopping worker: {e}")
        
        # Then fully close workers
        for worker in workers:
            try:
                worker.close()
            except Exception as e:
                print(f"Error closing worker: {e}")
        
        # Safely stop display manager with additional error handling
        if display_manager:
            try:
                # Ensure running flag is set to False before trying to stop
                display_manager.running = False
                display_manager.stop()
            except Exception as e:
                print(f"Error stopping display manager: {e}")
        
        # Stop the training dashboard
        if training_dashboard:
            try:
                training_dashboard.update_stats({'status': 'Completed'})
                training_dashboard.stop()
            except Exception as e:
                print(f"Error stopping training dashboard: {e}")
        
        # Save final model state if training was interrupted
        try:
            if episodes_completed > 0:
                interrupted_path = f'{model_dir}/tetris_ppo_interrupted.pth'
                torch.save({
                    'episodes_completed': episodes_completed,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'results': results
                }, interrupted_path)
                print(f"Saved interrupted training state to {interrupted_path}")
        except Exception as e:
            print(f"Error saving interrupted state: {e}")
            
        # Final cleanup
        torch.cuda.empty_cache()
        print("Cleanup complete.")
        
    return best_reward

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='Train PPO on Tetris')
        parser.add_argument('--num_episodes', type=int, default=1000)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--lam', type=float, default=0.95)
        parser.add_argument('--clip_ratio', type=float, default=0.2)
        parser.add_argument('--target_kl', type=float, default=0.01)
        parser.add_argument('--value_coef', type=float, default=0.5)
        parser.add_argument('--entropy_coef', type=float, default=0.1)
        parser.add_argument('--learning_rate', type=float, default=3e-4)
        parser.add_argument('--max_moves', type=int, default=100)
        parser.add_argument('--save_interval', type=int, default=100)
        parser.add_argument('--weights', type=str, default=None)
        parser.add_argument('--level', type=int, default=1)
        parser.add_argument('--level_inc', type=int, default=-1)
        parser.add_argument('--cycles', type=int, default=1)
        parser.add_argument('--debug', action='store_true', help='Enable debugging')
        parser.add_argument('--no_display', dest='display_enabled', action='store_false', default=True)
        parser.add_argument('--record', action='store_true', help='Record video')
        parser.add_argument('--num_workers', type=int, default=1, help='Number of parallel workers')
        parser.add_argument('--display_worker', type=int, default=0, help='Worker ID to display (must be < num_workers)')
        parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file for resuming')
        parser.add_argument('--stream', action='store_true', help='Enable streaming on local ports')
        parser.add_argument('--stream_base_port', type=int, default=8080, help='Base port for streaming (if enabled)')
        
        args = parser.parse_args()
        
        # Validate arguments
        if args.display_worker >= args.num_workers:
            print(f"Warning: display_worker {args.display_worker} >= num_workers {args.num_workers}, setting to 0")
            args.display_worker = 0
            
        # Set streaming port if enabled
        stream_base_port = args.stream_base_port if args.stream else None
        
        print(f"Arguments: {args}")
        main(
            num_episodes=args.num_episodes,
            batch_size=args.batch_size,
            gamma=args.gamma,
            lam=args.lam,
            clip_ratio=args.clip_ratio,
            target_kl=args.target_kl,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            learning_rate=args.learning_rate,
            max_moves=args.max_moves,
            save_interval=args.save_interval,
            weights=args.weights,
            display_enabled=args.display_enabled,
            record=args.record,
            level=args.level,
            level_inc=args.level_inc,
            cycles=args.cycles,
            debug=args.debug,
            num_workers=args.num_workers,
            display_worker=args.display_worker,
            checkpoint=args.checkpoint,
            stream_base_port=stream_base_port
        )
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
        # Ensure any resources at the top level are cleaned up
        torch.cuda.empty_cache()