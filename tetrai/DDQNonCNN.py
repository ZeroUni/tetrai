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

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class FrameStack:
    def __init__(self, stack_size):
        self.stack_size = stack_size
        self.frames = deque([], maxlen=stack_size)

    def __call__(self, frame):
        if len(self.frames) == 0:
            for _ in range(self.stack_size):
                self.frames.append(frame)
        else:
            self.frames.append(frame)

        # Proper stacking for single/batch inputs
        if len(frame.shape) == 3:  # Single frame [C,H,W]
            stacked = torch.stack(list(self.frames), dim=0)  # [stack_size,C,H,W]
        else:  # Batch of frames [B,C,H,W]
            stacked = torch.stack(list(self.frames), dim=1)  # [B,stack_size,C,H,W]
            stacked = stacked.squeeze(2)  # Remove extra channel dim
        return stacked

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """return a random batch of samples from our memory"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
class PrioritizedReplayMemory(ReplayMemory):
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        super().__init__(capacity)
        self.priorities = deque([], maxlen=capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.eps = 1e-6

    def push(self, *args):
        # Convert any tensors to scalar values for priorities
        max_priority = float(max([p.item() if torch.is_tensor(p) else float(p) 
                                for p in self.priorities])) if self.priorities else 1.0
        
        self.memory.append(Transition(*args))
        self.priorities.append(max_priority)

    def sample(self, batch_size):
        if len(self.memory) == 0:
            return None, None, None
        
        # Convert priorities to numpy array safely
        priorities = np.array([p.item() if torch.is_tensor(p) else float(p) 
                            for p in self.priorities])
        
        # Ensure no negative or zero priorities
        priorities = np.maximum(priorities, self.eps)
        
        # Calculate probabilities with numeric stability
        probs = priorities ** self.alpha
        probs_sum = probs.sum()
        
        # Handle case where sum is zero or invalid
        if not np.isfinite(probs_sum) or probs_sum <= 0:
            # Fall back to uniform distribution
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs_sum
        
        # Verify probabilities are valid
        assert np.all(np.isfinite(probs))
        assert np.abs(np.sum(probs) - 1.0) < 1e-5
        
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32, device='cuda')
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            if isinstance(priority, np.ndarray):
                if priority.size > 1:
                    p = float(priority.mean())  # Take mean if array has multiple values
                else:
                    p = float(priority.item())
            elif torch.is_tensor(priority):
                if priority.numel() > 1:
                    p = float(priority.mean().item())
                else:
                    p = float(priority.item())
            else:
                p = float(priority)
            self.priorities[idx] = p + self.eps

class EpisodeBuffer:
    def __init__(self, gamma=0.99, n_step=3):
        self.states = []
        self.actions = []
        self.immediate_rewards = []
        self.next_states = []
        self.dones = []
        self.gamma = gamma
        self.n_step = n_step
        
    def push(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.immediate_rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def calculate_n_step_returns(self):
        returns = []
        for t in range(len(self.immediate_rewards)):
            n_step_return = 0
            for i in range(self.n_step):
                if t + i < len(self.immediate_rewards):
                    n_step_return += (self.gamma ** i) * self.immediate_rewards[t + i]
            returns.append(n_step_return)
        return returns
    
    def get_transitions(self, memory):
        returns = self.calculate_n_step_returns()
        
        # Calculate final episode metrics
        final_score = sum(self.immediate_rewards)
        
        # Apply retrospective rewards
        for t in range(len(returns)):
            state = self.states[t]
            action = self.actions[t]
            next_state = self.next_states[t]
            done = self.dones[t]
            
            # Composite reward combining:
            # - N-step return
            # - Final episode outcome
            reward = (
                0.7 * returns[t] +  # n-step return
                0.3 * final_score * (t / len(returns))  # weighted final score
            )
            
            memory.push(state, action, reward, next_state, done)

def preprocess_state_batch(states, frame_stack=None):
    if not isinstance(states, list):
        states = [states]
        
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    # Process frames
    processed = torch.stack([preprocess(state) for state in states])
    
    if frame_stack:
        processed = frame_stack(processed)
        processed = processed.reshape(-1, 4, 128, 128)  # Ensure correct shape
    
    # Normalize and move to GPU
    processed = ((processed - 0.5) / 0.5)
    processed = processed.contiguous(memory_format=torch.channels_last)
    processed = processed.to(device='cuda', non_blocking=True)
    
    return processed

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
    display_enabled=True
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
    
    policy_net = TetrisCNN.TetrisCNN(num_actions, device=device).to(memory_format=torch.channels_last)
    target_net = TetrisCNN.TetrisCNN(num_actions, device=device).to(memory_format=torch.channels_last)
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
    patience = 50
    best_reward = -1 # Since we want to focus on score, we can assume it will never go below 0 on its own
    no_improve = 0

    steps_done = 0

    if resume:
        with open(policy_net_path, 'rb') as f:
            policy_net.load_state_dict(torch.load(f))
        with open(target_net_path, 'rb') as f:
            target_net.load_state_dict(torch.load(f))

    # Start training?
    try:
        with torch.cuda.device(device):
            for episode in range(num_episodes):
                # Apply schedulers
                if episode < warmup_steps:
                    schedulers['warmup'].step()
                else:
                    schedulers['cyclic'].step()

                episode_losses = []
                episode_buffer = EpisodeBuffer(gamma=gamma)
                
                with torch.cuda.stream(compute_stream):
                    state = env.reset()
                    state = preprocess_state_batch(state, frame_stack).to(device)
                    total_reward = 0
                    done = False

                    while not done:
                        # Temperature-scaled exploration
                        with autocast(device_type='cuda', dtype=torch.float16):
                            state = state.reshape(-1, 4, 128, 128)
                            q_values = policy_net(state) / temperature
                            
                            # Epsilon-greedy with legal action masking
                            if random.random() < epsilon:
                                legal_actions = env.get_legal_actions()
                                action = random.choice(legal_actions)
                            else:
                                # Mask illegal actions with large negative values
                                action_mask = torch.tensor(env.get_legal_actions_mask(), 
                                                        device=device, 
                                                        dtype=torch.float16)
                                masked_q_values = q_values + (action_mask - 1) * 1e9
                                action = masked_q_values.argmax(dim=1).item()

                        # Perform action
                        next_state, reward, done = env.step(action)
                        if reward != 0:
                            reward = reward_normalizer.normalize(torch.tensor(reward, device=device)).item()

                        with torch.cuda.stream(memory_stream):
                            next_state = preprocess_state_batch(next_state, frame_stack)
                            next_state = next_state.to(device)

                            # Store transition in episode buffer instead of memory
                            episode_buffer.push(state, action, reward, next_state, done)

                        memory_stream.synchronize()
                        total_reward += reward
                        state = next_state
                        steps_done += 1

                        # Update exploration parameters
                        epsilon = max(epsilon_end, epsilon * epsilon_decay)
                        temperature = max(0.1, temperature * temp_decay)

                # Process episode buffer and update memory after episode completion
                episode_buffer.get_transitions(memory)

                # Training phase - only after episode completion
                if len(memory) >= batch_size:
                    for _ in range(4):  # Multiple training iterations per episode
                        batch, indices, weights = memory.sample(batch_size)
                        if batch is None:
                            continue
                            
                        transitions = batch
                        batch = Transition(*zip(*transitions))

                        # Prepare batches efficiently
                        with torch.cuda.stream(memory_stream):
                            state_batch = torch.stack(batch.state).reshape(-1, 4, 128, 128)
                            action_batch = torch.tensor(batch.action, device=device).unsqueeze(1)
                            reward_batch = torch.tensor(batch.reward, device=device)
                            next_state_batch = torch.stack(batch.next_state).reshape(-1, 4, 128, 128)
                            done_batch = torch.tensor(batch.done, device=device, dtype=torch.float16)

                        memory_stream.synchronize()

                        # Training step with mixed precision
                        with autocast(device_type='cuda', dtype=torch.float16):
                            # Double Q-learning with temperature scaling
                            next_q_values = policy_net(next_state_batch) / temperature
                            next_actions = next_q_values.argmax(dim=1, keepdim=True)
                            next_target_values = target_net(next_state_batch).gather(1, next_actions)
                            
                            # Enhanced TD error calculation
                            expected_q_values = reward_batch + (gamma * next_target_values * (1 - done_batch))
                            current_q_values = policy_net(state_batch).gather(1, action_batch)
                            
                            # Huber loss for stability
                            td_errors = F.smooth_l1_loss(current_q_values, expected_q_values.detach(), 
                                                        reduction='none')
                            loss = (weights * td_errors).mean()

                        # Gradient clipping and scaling
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)

                        episode_losses.append(loss.item())
                        
                        # Update priorities
                        new_priorities = td_errors.detach().cpu().numpy()
                        memory.update_priorities(indices, new_priorities)

                # Track metrics
                optimizer.step()
                policy_net.reset_noise()
                target_net.reset_noise()
                
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
        cleanup(env, render_queue, display_thr, display_enabled)
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
        weights=args.weights
    )