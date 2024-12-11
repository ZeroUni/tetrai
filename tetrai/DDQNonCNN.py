import torch
from torch.amp import autocast, GradScaler
import torch.cuda.amp as amp
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import torch.nn.functional as F

import torchvision.transforms as transforms

import argparse

from collections import namedtuple, deque
import traceback

import numpy as np

from PIL import Image
import time
import os

import threading
import multiprocessing
import cv2

import pygame

import random

import TetrisCNN
import TetrisEnv
from rewardNormalizer import RewardNormalizer

import json
from torch.profiler import profile, record_function, ProfilerActivity
from contextlib import nullcontext

import zlib
import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class CompressedTensor:
    def __init__(self, tensor, compression_level=1):
        # Ensure tensor is in valid range [-1, 1]
        tensor = torch.clamp(tensor, -1.0, 1.0)
        # Scale to [-127, 127] range for int8
        np_data = (tensor.cpu().numpy() * 127).clip(-127, 127).astype(np.int8)
        # Compress the data
        self.compressed_data = zlib.compress(np_data.tobytes(), level=compression_level)
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.device = tensor.device

    def decompress(self):
        # Decompress and convert back to tensor
        np_data = np.frombuffer(zlib.decompress(self.compressed_data), dtype=np.int8)
        np_data = np_data.reshape(self.shape)
        # Convert back to [-1, 1] range
        tensor = torch.from_numpy(np.array(np_data, copy=True)).float().to(self.device) / 127.0
        return tensor.to(dtype=self.dtype)

class FrameStack:
    def __init__(self, stack_size):
        self.stack_size = stack_size
        self.frames = deque([], maxlen=stack_size)
        self.batch_frames = None
        self.compressed_frames = deque([], maxlen=stack_size)
        self.compression_level = 1
        
    def reset(self):
        self.compressed_frames.clear()
        self.batch_frames = None
        torch.cuda.empty_cache()

    @torch.no_grad()
    def __call__(self, frame):
        if isinstance(frame, list):  # Batch processing
            return self._process_batch(frame)
        return self._process_single(frame)

    def _process_single(self, frame):
        if len(self.compressed_frames) == 0:
            for _ in range(self.stack_size):
                compressed = CompressedTensor(frame, self.compression_level)
                self.compressed_frames.append(compressed)
        else:
            self.compressed_frames.popleft()
            compressed = CompressedTensor(frame, self.compression_level)
            self.compressed_frames.append(compressed)

        # Decompress only when needed
        decompressed = [f.decompress() for f in self.compressed_frames]
        stacked = torch.stack(decompressed, dim=0)
        return stacked.to('cuda', non_blocking=True)

    def _process_batch(self, frames):
        batch_size = len(frames)
        if self.batch_frames is None:
            self.batch_frames = deque([
                torch.stack([f.clone().detach_() for f in frames])
                for _ in range(self.stack_size)
            ], maxlen=self.stack_size)
        else:
            old_batch = self.batch_frames.popleft()
            del old_batch
            self.batch_frames.append(torch.stack([f.clone().detach_() for f in frames]))

        stacked = torch.stack(list(self.batch_frames), dim=1)
        return stacked.to('cuda', non_blocking=True)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.device = 'cuda'
        self.compression_level = 1

    def push(self, *args):
        # Compress state tensors before storage
        state = CompressedTensor(args[0], self.compression_level)
        next_state = CompressedTensor(args[3], self.compression_level)
        
        # Clean up old tensors
        if len(self.memory) >= self.memory.maxlen:
            old_transition = self.memory[0]
            del old_transition

        self.memory.append(Transition(state, args[1], args[2], next_state, args[4]))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        # Decompress states during sampling
        decompressed = []
        for t in transitions:
            state = t.state.decompress()
            next_state = t.next_state.decompress()
            decompressed.append(Transition(state, t.action, t.reward, next_state, t.done))
        return decompressed
    
    def __len__(self):
        return len(self.memory)
    
import threading
class PrioritizedReplayMemory(ReplayMemory):
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        super().__init__(capacity)
        self.priorities = deque([], maxlen=capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.eps = 1e-6
        self.lock = threading.Lock()  # Add thread safety

    def push(self, *args):
        with self.lock:
            # Clean up old tensors
            if len(self.memory) >= self.memory.maxlen:
                old_transition = self.memory[0]
                if hasattr(old_transition.state, 'detach'):
                    old_transition.state.detach_()
                if hasattr(old_transition.next_state, 'detach'):
                    old_transition.next_state.detach_()

            # Set initial priority to max priority seen so far
            max_priority = max(self.priorities) if self.priorities else 1.0
            
            self.memory.append(Transition(*args))
            self.priorities.append(max_priority)

    def sample(self, batch_size):
        with self.lock:
            if len(self.memory) == 0:
                return None, None, None
            
            if len(self.memory) < batch_size:
                batch_size = len(self.memory)

            # Create local copy of priorities
            priorities = np.array([p for p in self.priorities])
            
            # Ensure arrays have same size
            if len(priorities) > len(self.memory):
                priorities = priorities[:len(self.memory)]
            elif len(priorities) < len(self.memory):
                padding = np.ones(len(self.memory) - len(priorities))
                priorities = np.concatenate([priorities, padding])

            # Calculate sampling probabilities
            priorities = np.array(priorities, dtype=np.float64)  # Use float64 for numerical stability
            priorities = np.maximum(priorities, self.eps)
            probs = priorities ** self.alpha
            probs = probs / np.sum(probs)  # Normalize

            # Sample indices and get transitions
            try:
                indices = np.random.choice(len(self.memory), batch_size, p=probs)
            except ValueError as e:
                print(f"Error sampling: probs sum={np.sum(probs)}, len(memory)={len(self.memory)}, len(probs)={len(probs)}")
                raise e

            samples = [self.memory[idx] for idx in indices]
            
            # Calculate importance sampling weights
            total = len(self.memory)
            weights = (total * probs[indices]) ** (-self.beta)
            weights = torch.tensor(weights / weights.max(), device='cuda', dtype=torch.float32)
            
            self.beta = min(1.0, self.beta + self.beta_increment)
            
            return samples, indices, weights

    def update_priorities(self, indices, priorities):
        with self.lock:
            # Handle scalar input
            if isinstance(priorities, (float, int)):
                priorities = np.array([priorities])
            elif torch.is_tensor(priorities):
                if priorities.ndim == 0:
                    priorities = priorities.unsqueeze(0)
                priorities = priorities.cpu().numpy()
            
            # Ensure indices is an array
            if isinstance(indices, (int, np.integer)):
                indices = [indices]
            
            # Update priorities with bounds checking
            for idx, priority in zip(indices, priorities):
                if idx < len(self.priorities):
                    self.priorities[idx] = float(priority) + self.eps

class EpisodeBuffer:
    def __init__(self, gamma=0.99, n_step=3):
        self.device = 'cuda'
        self.max_size = 1000
        # Store references to compressed states instead of tensors
        self.states = []
        self.actions = []
        self.immediate_rewards = []
        self.next_states = []
        self.dones = []
        self.current_idx = 0
        
        self.gamma = gamma
        self.n_step = n_step
        self.gamma_powers = torch.tensor([gamma ** i for i in range(n_step)], 
                                       device=self.device)
        self.compression_level = 1

    @autocast(device_type='cuda', dtype=torch.float16)
    def push(self, state, action, reward, next_state, done):
        if self.current_idx >= self.max_size:
            return
            
        with torch.cuda.stream(torch.cuda.Stream()):
            # Compress states
            compressed_state = CompressedTensor(state, self.compression_level)
            compressed_next_state = CompressedTensor(next_state, self.compression_level)
            
            # Append to lists instead of tensor assignment
            self.states.append(compressed_state)
            self.next_states.append(compressed_next_state)
            self.actions.append(action)
            self.immediate_rewards.append(reward)
            self.dones.append(done)
            self.current_idx += 1

    @autocast(device_type='cuda', dtype=torch.float16)
    def calculate_n_step_returns(self):
        episode_length = self.current_idx
        if episode_length == 0:
            return torch.empty(0, device=self.device)

        # Convert rewards to tensor with explicit dtype
        rewards = torch.tensor(self.immediate_rewards[:episode_length], 
                             device=self.device, 
                             dtype=torch.float32)  # Change to float32
        
        # Initialize returns with same dtype
        returns = torch.zeros_like(rewards, dtype=torch.float32)
        
        # Calculate n-step returns with explicit casting
        for i in range(min(self.n_step, episode_length)):
            gamma_power = float(self.gamma ** i)  # Explicit float conversion
            shifted_rewards = torch.cat([
                rewards[i:],
                torch.zeros(i, device=self.device, dtype=torch.float32)
            ])
            returns += gamma_power * shifted_rewards[:episode_length]
        
        return returns

    def get_transitions(self, memory):
        episode_length = self.current_idx
        if episode_length == 0:
            return

        with torch.cuda.stream(torch.cuda.Stream()):
            returns = self.calculate_n_step_returns()
            final_score = torch.tensor(sum(self.immediate_rewards[:episode_length]), 
                                     device=self.device, 
                                     dtype=torch.float32)  # Explicit dtype
            
            progress_weights = torch.linspace(0, 1, episode_length, 
                                           device=self.device, 
                                           dtype=torch.float32)  # Explicit dtype
            rewards = (0.7 * returns + 0.3 * final_score * progress_weights)

            batch_size = 128
            for i in range(0, episode_length, batch_size):
                end_idx = min(i + batch_size, episode_length)
                batch_slice = slice(i, end_idx)
                
                # Get batch of transitions
                states = [self.states[j].decompress() for j in range(i, end_idx)]
                actions = self.actions[i:end_idx]
                batch_rewards = rewards[i:end_idx]
                next_states = [self.next_states[j].decompress() for j in range(i, end_idx)]
                dones = self.dones[i:end_idx]
                
                transitions = zip(
                    states,
                    actions,
                    batch_rewards.tolist(),
                    next_states,
                    dones
                )
                
                for transition in transitions:
                    memory.push(*transition)

        self._clear_buffers()

    def _clear_buffers(self):
        self.current_idx = 0
        self.states.clear()
        self.actions.clear()
        self.immediate_rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        torch.cuda.empty_cache()

class PreprocessingPool:
    def __init__(self, size=32, shape=(1, 128, 128)):
        self.size = size
        self.shape = shape
        self.pool = torch.zeros((size, *shape), 
                              dtype=torch.float16, 
                              device='cuda',
                              pin_memory=True)
        self.available = set(range(size))
        self.in_use = {}  # Change to dict to track tensor references
        self.lock = threading.Lock()

    @torch.no_grad()
    def get_tensor(self, shape=None):
        with self.lock:
            if not self.available:
                # Add warning when pool is exhausted
                print("Warning: PreprocessingPool exhausted, creating temporary tensor")
                return torch.zeros(shape or self.shape, 
                                 dtype=torch.float16, 
                                 device='cuda')
            
            idx = self.available.pop()
            tensor = self.pool[idx]
            tensor.zero_()
            # Store tensor reference and index
            self.in_use[tensor.data_ptr()] = idx
            return tensor

    def return_tensor(self, tensor):
        with self.lock:
            if tensor is None:
                return
                
            ptr = tensor.data_ptr()
            if ptr in self.in_use:
                idx = self.in_use.pop(ptr)
                self.available.add(idx)
                # Clear the tensor
                self.pool[idx].zero_()
            # Else silently ignore tensors not from our pool

    def reset(self):
        with self.lock:
            self.available = set(range(self.size))
            self.in_use.clear()
            self.pool.zero_()
            torch.cuda.empty_cache()

    def __del__(self):
        # Ensure cleanup on deletion
        self.reset()
        del self.pool

# Update preprocess_state_batch to handle errors better
@torch.no_grad()
def preprocess_state_batch(states, frame_stack=None, device='cuda', preprocess_pool=None):
    if not isinstance(states, list):
        states = [states]
        
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    processed_states = []
    tensors_to_return = []
    try:
        with autocast(device_type=str(device), dtype=torch.float16):
            for state in states:
                try:
                    # Get tensor from pool
                    processed = preprocess_pool.get_tensor() if preprocess_pool else \
                               torch.zeros((1, 128, 128), dtype=torch.float16, device=device)
                    
                    if preprocess_pool:
                        tensors_to_return.append(processed)
                    
                    # Process directly into pooled tensor
                    processed.copy_(preprocess(state).to(device, non_blocking=True))
                    processed_states.append(processed)
                except Exception as e:
                    print(f"Error processing state: {e}")
                    # Return any acquired tensors
                    for tensor in tensors_to_return:
                        preprocess_pool.return_tensor(tensor)
                    raise e
                
            # Stack tensors
            processed = torch.stack(processed_states)
            processed = processed.to(memory_format=torch.channels_last)
            
            if frame_stack:
                processed = frame_stack(processed)
                processed = processed.reshape(-1, 4, 128, 128)
                processed = processed.to(memory_format=torch.channels_last)
            
            # Normalize in-place
            processed.sub_(0.5).div_(0.5)
            
            return processed
            
    finally:
        # Always return tensors to pool
        if preprocess_pool:
            for tensor in tensors_to_return:
                preprocess_pool.return_tensor(tensor)

def render_thread(env):
    while True:
        env.render()
        pygame.time.wait(env.render_delay)

def display_images(render_queue):
    # Create the display window
    tag = os.getpid()
    window_name = f'Tetris_{tag}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 512, 512)

    while True:
        try:
            image = render_queue.get()
            if image is None:
                break
            # Convert RGB to BGR for OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(e)
            traceback.print_exc()
            break
    cv2.destroyWindow(window_name)

def cleanup(env, render_queue=None, display_thr=None, display_enabled=False):
    env.close()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    if display_enabled and render_queue and display_thr:
        render_queue.put(None)
        display_thr.join()

import threading
from queue import Queue
import torch.multiprocessing as mp

class AsyncBufferProcessor:
    def __init__(self, memory, device='cuda'):
        self.memory = memory
        self.device = device
        self.queue = Queue()
        self.processing_thread = threading.Thread(target=self._process_buffer, daemon=True)
        self.processing_thread.start()
        self.current_buffer = EpisodeBuffer(gamma=0.99)
        self.next_buffer = EpisodeBuffer(gamma=0.99)
        self.lock = threading.Lock()  # Add thread safety

    def _process_buffer(self):
        while True:
            buffer = self.queue.get()
            if buffer is None:
                break
            
            with torch.cuda.stream(torch.cuda.Stream()):
                buffer.get_transitions(self.memory)
                del buffer

    def push(self, state, action, reward, next_state, done):
        with self.lock:
            self.current_buffer.push(state, action, reward, next_state, done)

    def swap_buffers(self):
        with self.lock:
            # Queue current buffer for processing and swap to next buffer
            if self.current_buffer.current_idx > 0:  # Only process non-empty buffers
                self.queue.put(self.current_buffer)
                self.current_buffer, self.next_buffer = self.next_buffer, EpisodeBuffer(gamma=0.99)

    def cleanup(self):
        self.queue.put(None)
        self.processing_thread.join()

def main(
    resume=False,
    num_episodes=1000,
    batch_size=128,
    gamma=0.99,
    target_update=5,
    memory_capacity=10000,
    learning_rate=1e-5,
    policy_net_path='tetris_policy_net.pth',
    target_net_path='tetris_target_net.pth',
    max_moves=100,
    save_interval=500,
    weights=None,
    display_enabled=True,
    debug=False
):
    # Define Hyperparameters
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    compute_stream = torch.cuda.Stream()
    memory_stream = torch.cuda.Stream()

    num_actions = 8 # Nothing, rotate left / right, move down / left / right, place down, hold

    # get a unique name for our model dir based on the time
    model_dir = f'out/{int(time.time())}'
    os.makedirs(model_dir, exist_ok=True)

    # Initialize our networks and torch optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    policy_net = TetrisCNN.RainbowTetrisCNN(
        num_actions=8,
        device=device,
        num_atoms=51,
        v_min=-10,
        v_max=10
    ).to(device, memory_format=torch.channels_last)

    target_net = TetrisCNN.RainbowTetrisCNN(
        num_actions=8,
        device=device,
        num_atoms=51,
        v_min=-10,
        v_max=10
    ).to(device, memory_format=torch.channels_last)
    
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    reward_normalizer = RewardNormalizer()
    
    if display_enabled:
        render_queue = multiprocessing.Queue()
        display_thr = threading.Thread(target=display_images, args=(render_queue,))
        display_thr.start()
    else:
        display_thr = None
        render_queue = None
    
    # Load the weights from json if provided
    if weights:
        if isinstance(weights, str): # Load from file
            with open(weights, 'r') as f:
                weights = json.load(f)
        elif isinstance(weights, dict): # Use directly
            pass


    env = TetrisEnv.TetrisEnv(render_queue, max_moves=max_moves, weights=weights)
    env.render_mode = False

    # Enhanced optimizer with weight decay
    optimizer = optim.AdamW(
        policy_net.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Multi-stage learning rate scheduler
    warmup_steps = num_episodes // 10
    plateau_steps = num_episodes // 5
    
    schedulers = {
        'warmup': optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        ),
        'cyclic': optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=learning_rate,
            max_lr=learning_rate * 10,
            step_size_up=plateau_steps,
            mode='triangular2',
            cycle_momentum=False
        )
    }

    # Temperature scaling for exploration
    temperature = 1.0
    temp_decay = 0.995

    # Epsilon greedy parameters
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    epsilon = epsilon_start

    memory = PrioritizedReplayMemory(memory_capacity)
    frame_stack = FrameStack(stack_size=4)
    scaler = GradScaler()

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    state_batch = torch.zeros(
        (batch_size, 4, 128, 128),
        dtype=torch.float16,
        device=device,
        requires_grad=False
    ).to(memory_format=torch.channels_last)

    # Initialize results tracking
    results = {
        'episode_rewards': [],
        'avg_losses': [],
        'learning_rates': [],
        'epsilon': [],
        'temperature': []
    }

    # Early stopping parameters
    patience = 40
    best_reward = -1 # Since we want to focus on score, we can assume it will never go below 0 on its own
    no_improve = 0

    steps_done = 0

    if resume:
        with open(policy_net_path, 'rb') as f:
            policy_net.load_state_dict(torch.load(f))
        with open(target_net_path, 'rb') as f:
            target_net.load_state_dict(torch.load(f))
    # Start training?
    buffer_processor = AsyncBufferProcessor(memory, device)
    training_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()
    
    try:
        # Create context manager for profiler if debug is True
        profiler_ctx = (
            torch.profiler.profile(
                activities=[
                    ProfilerActivity.CPU,
                    ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=3
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/tetris_profiler'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) if debug else nullcontext()
        )

        with profiler_ctx as prof:
            with torch.cuda.device(device):
                for episode in range(num_episodes):
                    with record_function("episode") if debug else nullcontext():
                        episode_losses = []
                        
                        # Episode execution in compute stream
                        with torch.cuda.stream(compute_stream):
                            state = env.reset()
                            state = preprocess_state_batch(state, frame_stack).to(device)
                            total_reward = 0
                            done = False

                            while not done:
                                with record_function("action_selection") if debug else nullcontext():
                                    # Temperature-scaled exploration
                                    with autocast(device_type=str(device), dtype=torch.float16):
                                        state = state.reshape(-1, 4, 128, 128)
                                        q_values = policy_net(state) / temperature
                                        
                                        # Epsilon-greedy with legal action masking
                                        if random.random() < epsilon:
                                            legal_actions = env.get_legal_actions()
                                            action = random.choice(legal_actions)
                                        else:
                                            with torch.no_grad():
                                                # Get Q-values by computing expected values from distributions
                                                q_dist = policy_net(state) # Shape: [batch, num_actions, num_atoms]
                                                q_values = (q_dist * policy_net.supports).sum(dim=-1) # Shape: [batch, num_actions]
                                                
                                                # Apply action masking
                                                action_mask = env.get_legal_actions_mask().to(device)
                                                masked_q_values = q_values + (action_mask - 1) * 1e9
                                                action = masked_q_values.argmax(dim=1).item()

                                with record_function("environment_step") if debug else nullcontext():
                                    # Perform action
                                    next_state, reward, done = env.step(action)
                                    if reward != 0:
                                        reward = reward_normalizer.normalize(torch.tensor(reward, device=device)).item()

                                with record_function("state_processing") if debug else nullcontext():
                                    with torch.cuda.stream(memory_stream):
                                        next_state = preprocess_state_batch(next_state, frame_stack)
                                        next_state = next_state.to(device)

                                        # Store transition in episode buffer instead of memory
                                        buffer_processor.push(state, action, reward, next_state, done)

                                memory_stream.synchronize()
                                total_reward += reward
                                state = next_state
                                steps_done += 1

                                # Update exploration parameters
                                epsilon = max(epsilon_end, epsilon * epsilon_decay)
                                temperature = max(0.1, temperature * temp_decay)

                    # Swap buffers asynchronously
                    buffer_processor.swap_buffers()

                    # Training phase with async prefetch
                    if len(memory) >= batch_size:
                        with torch.cuda.stream(training_stream):
                            # Prefetch next batch while processing current
                            next_batch = None
                            for _ in range(4):
                                current_batch = next_batch if next_batch else memory.sample(batch_size)
                                next_batch = memory.sample(batch_size)
                                
                                if current_batch[0] is None:
                                    continue

                                batch, indices, weights = current_batch
                                transitions = batch
                                batch = Transition(*zip(*transitions))

                                # Prepare batches with prefetching
                                state_batch = torch.stack([s for s in batch.state])
                                action_batch = torch.tensor(batch.action, device=device)
                                reward_batch = torch.tensor(batch.reward, device=device)
                                next_state_batch = torch.stack([s for s in batch.next_state])
                                done_batch = torch.tensor(batch.done, device=device, dtype=torch.float16)

                                # Reshape tensors once
                                state_batch = state_batch.reshape(-1, 4, 128, 128)
                                next_state_batch = next_state_batch.reshape(-1, 4, 128, 128)

                                # Forward pass
                                with autocast(device_type=str(device), dtype=torch.float16):
                                    # Get current state distribution
                                    dist = policy_net(state_batch, mode='dqn')
                                    action_dist = dist[range(batch_size), action_batch.squeeze()]

                                    # Get next state distribution
                                    with torch.no_grad():
                                        next_dist = target_net(next_state_batch, mode='dqn')

                                        # Calculate expected Q-values
                                        next_q = (next_dist * policy_net.supports).sum(-1) 
                                        next_actions = next_q.argmax(1)

                                        next_dist = next_dist[range(batch_size), next_actions]

                                        # Properly reshape tensors for broadcasting
                                        reward_batch = reward_batch.unsqueeze(-1)  # [B, 1]
                                        done_batch = done_batch.unsqueeze(-1)      # [B, 1]
                                        supports = policy_net.supports.unsqueeze(0) # [1, num_atoms]

                                        # Compute projected distribution with proper broadcasting
                                        projected_atoms = reward_batch + (1 - done_batch) * gamma * supports
                                        projected_atoms.clamp_(policy_net.v_min, policy_net.v_max)
                                        
                                        # Project onto support
                                        delta_z = policy_net.delta_z
                                        bj = (projected_atoms - policy_net.v_min) / delta_z
                                        l = bj.floor().long()  # Shape: [batch_size, num_atoms]
                                        u = bj.ceil().long()   # Shape: [batch_size, num_atoms]

                                        # Handle corner cases
                                        l[(u > 0) * (l == u)] -= 1
                                        u[(l < (policy_net.num_atoms - 1)) * (l == u)] += 1

                                        # Create projected distribution more efficiently
                                        proj_dist = torch.zeros_like(action_dist)
                                        for i in range(batch_size):
                                            for j in range(policy_net.num_atoms):
                                                l_idx = l[i, j]
                                                u_idx = u[i, j]
                                                prob = next_dist[i, j]
                                                
                                                proj_dist[i, l_idx] += prob * (u_idx.float() - bj[i, j])
                                                proj_dist[i, u_idx] += prob * (bj[i, j] - l_idx.float())

                                    # Calculate KL divergence loss per sample
                                    losses = -(proj_dist * torch.log(action_dist + 1e-8)).sum(-1)
                                    # Calculate mean loss for backward pass
                                    loss = (weights * losses).mean()

                                    # Update priorities using per-sample losses
                                    memory.update_priorities(indices, losses.detach())

                                # Gradient clipping and scaling
                                scaler.scale(loss).backward()
                                scaler.unscale_(optimizer)
                                clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
                                scaler.step(optimizer)
                                scaler.update()
                                optimizer.zero_grad(set_to_none=True)

                                episode_losses.append(loss.item())
                                
                                # Remove redundant priority update
                                # memory.update_priorities(indices, new_priorities)  # Remove this line

                        # Synchronize streams before next iteration
                        torch.cuda.synchronize()

                        # Explicit cleanup after each training step
                        del state_batch, action_batch, reward_batch, next_state_batch, done_batch
                        torch.cuda.empty_cache()

                    # Periodic memory cleanup
                    if episode % 5 == 0:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()

                    # Update networks and learning rate - moved after training
                    optimizer.step()
                    policy_net.reset_noise()
                    target_net.reset_noise()

                    # Apply schedulers after optimizer step
                    if episode < warmup_steps:
                        schedulers['warmup'].step()
                    else:
                        schedulers['cyclic'].step()

                    # Track metrics
                    avg_loss = np.mean(episode_losses) if episode_losses else 0
                    results['episode_rewards'].append(total_reward)
                    results['avg_losses'].append(avg_loss)
                    results['learning_rates'].append(schedulers['cyclic'].get_last_lr()[0])
                    results['epsilon'].append(epsilon)
                    results['temperature'].append(temperature)

                    # Early stopping check
                    score = env.get_score()
                    if score > best_reward:
                        best_reward = score
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= patience:
                            print("Early stopping triggered")
                            break

                    # Update the target network
                    if episode % target_update == 0:
                        target_net.load_state_dict(policy_net.state_dict())

                    print(f"Episode {episode}: Total Reward = {total_reward}")
                    reward_normalizer.reset()

                    if episode != 0 and episode % save_interval == 0:
                        # Save model and current results
                        torch.save(policy_net.state_dict(), f'{model_dir}/tetris_policy_net_{episode}.pth')
                        torch.save(target_net.state_dict(), f'{model_dir}/tetris_target_net_{episode}.pth')
                        with open(f'{model_dir}/training_results.json', 'w') as f:
                            json.dump(results, f)

                    # Modify profiler related code
                    if debug:
                        if episode == 10:
                            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                            break
                        prof.step()

                    # After each episode, force garbage collection
                    if episode % 10 == 0:
                        torch.cuda.empty_cache()

        # Save final results and models
        with open(f'{model_dir}/training_results.json', 'w') as f:
            json.dump(results, f)

        torch.save(policy_net.state_dict(), f'{model_dir}/tetris_policy_net_final.pth')
        torch.save(target_net.state_dict(), f'{model_dir}/tetris_target_net_final.pth')

    except Exception as e:
        print(e)
        traceback.print_exc()
        cleanup(env, render_queue, display_thr, display_enabled)
    finally:
        buffer_processor.cleanup()
        # Ensure thorough cleanup
        cleanup(env, render_queue, display_thr, display_enabled)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return best_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DDQN on Tetris')
    parser.add_argument('--resume', type=bool, default=None)
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--target_update', type=int, default=10)
    parser.add_argument('--memory_capacity', type=int, default=10000)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--policy_net', type=str, default='tetris_policy_net.pth')
    parser.add_argument('--target_net', type=str, default='tetris_target_net.pth')
    parser.add_argument('--max_moves', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--debug', action='store_true', help='Enable profiling and debugging')

    args = parser.parse_args()
    main(
        resume=args.resume,
        num_episodes=args.num_episodes,
        batch_size=args.batch_size,
        gamma=args.gamma,
        target_update=args.target_update,
        memory_capacity=args.memory_capacity,
        learning_rate=args.learning_rate,
        policy_net_path=args.policy_net,
        target_net_path=args.target_net,
        max_moves=args.max_moves,
        save_interval=args.save_interval,
        weights=args.weights,
        debug=args.debug
    )