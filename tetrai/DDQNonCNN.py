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
    cv2.namedWindow('Tetris', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Tetris', 512, 512)

    while True:
        image = render_queue.get()
        if image is None:
            break
        # Convert RGB to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Tetris', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def cleanup(env, render_queue=None, display_proc=None, display_enabled=False):
    env.close()
    torch.cuda.empty_cache()
    if display_enabled and render_queue and display_proc:
        render_queue.put(None)
        display_proc.join()

def main(weights=None, num_episodes=1000, max_moves=-1, display_enabled=True):
    # Define Hyperparameters
    try:
        parser = argparse.ArgumentParser(description='Train DDQN on Tetris')
        parser.add_argument('--resume', type=bool, default=None)
        parser.add_argument('--num_episodes', type=int, default=num_episodes)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--target_update', type=int, default=10)
        parser.add_argument('--memory_capacity', type=int, default=10000)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--policy_net', type=str, default='tetris_policy_net.pth')
        parser.add_argument('--target_net', type=str, default='tetris_target_net.pth')
        parser.add_argument('--max_moves', type=int, default=max_moves)
        parser.add_argument('--save_interval', type=int, default=500)
        parser.add_argument('--weights', type=str, default=None)

        args = parser.parse_args()

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

        compute_stream = torch.cuda.Stream()
        memory_stream = torch.cuda.Stream()

        num_actions = 8 # Nothing, rotate left / right, move down / left / right, place down, hold
        batch_size = args.batch_size
        gamma = args.gamma
        target_update = args.target_update
        memory_capacity = args.memory_capacity
        num_episodes = args.num_episodes
        learning_rate = args.learning_rate

        # get a unique name for our model dir based on the time
        model_dir = f'out/{int(time.time())}'
        os.makedirs(model_dir, exist_ok=True)

        # Initialize our networks and torch optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {device}')
        
        policy_net = TetrisCNN.TetrisCNN(num_actions).to(device, memory_format=torch.channels_last)
        target_net = TetrisCNN.TetrisCNN(num_actions).to(device, memory_format=torch.channels_last)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        
        if display_enabled:
            render_queue = multiprocessing.Queue()
            display_proc = multiprocessing.Process(target=display_images, args=(render_queue,))
            display_proc.start()
        else:
            display_proc = None
            render_queue = None
        
        # Load the weights from json if provided
        if args.weights:
            with open(args.weights, 'r') as f:
                weights = json.load(f)

        env = TetrisEnv.TetrisEnv(render_queue, max_moves=args.max_moves, weights=weights)
        env.render_mode = False

        optimizer = optim.Adam(policy_net.parameters())
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        memory = PrioritizedReplayMemory(memory_capacity)
        frame_stack = FrameStack(stack_size=4)
        scaler = GradScaler()

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
            'learning_rates': []
        }

        # Early stopping parameters
        patience = 20
        best_reward = float('-inf')
        no_improve = 0

        steps_done = 0

        if args.resume:
            policy_net.load_state_dict(torch.load(args.policy_net))
            target_net.load_state_dict(torch.load(args.target_net))

        # Start training?
        with torch.cuda.device(device):
            for episode in range(num_episodes):
                episode_losses = []
                with torch.cuda.stream(compute_stream):
                    state = env.reset()
                    state = preprocess_state_batch(state, frame_stack).to(device)
                    total_reward = 0
                    done = False

                    while not done:
                        with autocast(device_type='cuda', dtype=torch.float16):
                            state = state.reshape(-1, 4, 128, 128)  # Ensure correct shape
                            action = policy_net(state).argmax(dim=1).item()

                        # Perform action
                        next_state, reward, done = env.step(action)
                        
                        with torch.cuda.stream(memory_stream):
                            next_state = preprocess_state_batch(next_state, frame_stack)

                        memory_stream.synchronize()
                        total_reward += reward

                        # Store transition in memory
                        memory.push(state, action, reward, next_state, done)
                        state = next_state
                        steps_done += 1

                        # Sample a batch and optimize model
                        if len(memory) >= batch_size:
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
                                next_actions = policy_net(next_state_batch).argmax(dim=1, keepdim=True)
                                next_q_values = target_net(next_state_batch).gather(1, next_actions)
                                expected_q_values = reward_batch + gamma * next_q_values * (1 - done_batch)
                                q_values = policy_net(state_batch).gather(1, action_batch)
                                
                                # Use weights in loss calculation
                                td_errors = torch.abs(q_values - expected_q_values.detach()).squeeze()
                                loss = (weights * (td_errors ** 2)).mean()

                            # Scaled backprop
                            scaler.scale(loss).backward()
                            
                            # Clip gradients for stability
                            scaler.unscale_(optimizer)
                            clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                            
                            scaler.step(optimizer)
                            optimizer.step()  # Add this line
                            scheduler.step()  # Move this after optimizer.step()
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)

                            # Reset the noise in the noisy layers
                            policy_net.reset_noise()
                            target_net.reset_noise()

                            episode_losses.append(loss.item())

                            # Update priorities
                            new_priorities = td_errors.detach().cpu().numpy()
                            memory.update_priorities(indices, new_priorities)

                # Track metrics
                avg_loss = np.mean(episode_losses) if episode_losses else 0
                results['episode_rewards'].append(total_reward)
                results['avg_losses'].append(avg_loss)
                results['learning_rates'].append(scheduler.get_last_lr()[0])

                # Early stopping check
                if total_reward > best_reward:
                    best_reward = total_reward
                    no_improve = 0
                    torch.save(policy_net.state_dict(), f'{model_dir}/best_model.pth')
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print("Early stopping triggered")
                        break

                # Update the target network
                if episode % target_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                print(f"Episode {episode}: Total Reward = {total_reward}")

                if episode != 0 and episode % args.save_interval == 0:
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
        cleanup(env, render_queue, display_proc, display_enabled)
    finally:
        cleanup(env, render_queue, display_proc, display_enabled)
        return best_reward


if __name__ == "__main__":
    main()