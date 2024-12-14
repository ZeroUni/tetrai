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
from trainingUtils import DisplayManager

import json
from torch.profiler import profile, record_function, ProfilerActivity
from contextlib import nullcontext

import zlib

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
class CompressedTensor:
    def __init__(self, tensor, compression_level=1):
        if tensor.dtype == torch.uint8:
            # Keep in [0,255] range for images
            np_data = tensor.cpu().numpy().astype(np.uint8)
            self.data_type_str = 'uint8'
        else:
            # For other tensors in [-1,1] range
            #print(f'Clamping tensor {tensor}')
            tensor = torch.clamp(tensor, -1.0, 1.0)
            np_data = (tensor.cpu().numpy() * 127).clip(-127, 127).astype(np.int8)
            self.data_type_str = 'int8'
            
        self.compressed_data = zlib.compress(np_data.tobytes(), level=compression_level)
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.device = tensor.device

    def decompress(self):
        dtype_map = {'uint8': np.uint8, 'int8': np.int8}
        np_data = np.frombuffer(zlib.decompress(self.compressed_data), 
                                dtype=dtype_map[self.data_type_str])
        np_data = np_data.reshape(self.shape)
        
        tensor = torch.from_numpy(np_data.copy())
        if self.data_type_str == 'int8':
            # Convert back to [-1,1] range for non-image data
            tensor = tensor.float() / 127.0
            
        return tensor.to(device=self.device, dtype=self.dtype)
    
class FrameStack:
    def __init__(self, stack_size):
        self.stack_size = stack_size
        self.frames = deque([], maxlen=stack_size)
        self.batch_frames = None
        self.compressed_frames = deque([], maxlen=stack_size)
        self.compression_level = 1
        # Adjust pool shape to match input tensor shape
        self._pool = torch.zeros((stack_size, 1, 128, 128), 
                               dtype=torch.float16).pin_memory()  # Pin on CPU
        self._pool = self._pool.to('cuda')  # Move to GPU after pinning
        
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
        
        if frame.dim() == 4:  # [B, C, H, W]
            frame = frame.squeeze(0)
        if frame.dim() == 3:  # [C, H, W]
            frame = frame.squeeze(0)
            
        if len(self.compressed_frames) == 0:
            # Initialize by copying first frame for better temporal context
            for i in range(self.stack_size - 1):
                compressed = CompressedTensor(frame.clone(), self.compression_level)
                self.compressed_frames.append(compressed)
                self._pool[i].copy_(frame, non_blocking=True)
                
            # Add actual frame as most recent
            compressed = CompressedTensor(frame, self.compression_level)
            self.compressed_frames.append(compressed)
            self._pool[-1].copy_(frame, non_blocking=True)
        else:
            # Rotate pool tensors instead of reallocating
            self._pool.roll(-1, dims=0)
            self._pool[-1].copy_(frame, non_blocking=True)
            self.compressed_frames.popleft()
            compressed = CompressedTensor(frame, self.compression_level)
            self.compressed_frames.append(compressed)

        # Add channel dimension back for return
        return self._pool.unsqueeze(1)

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
        self.capacity = capacity  # Store capacity explicitly
        self.memory = []  # Use list instead of deque for indexed access
        self.device = 'cuda'
        self.compression_level = 1
        
        # Priority related members
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.eps = 1e-6
        
        # Thread safety and batching
        self.lock = threading.Lock()
        self.priority_updates_indices = []
        self.priority_updates_values = []
        self.update_threshold = 32
        self.min_priority = 1e-6
        
        # State tracking
        self.state_hashes = {}
        self.max_hash_entries = capacity // 2
        self.hash_threshold = 0.95
        self.reward_boost_factor = 2.0
        self.uniqueness_boost = 1.5
        
        # Stability parameters
        self.max_priority = 1.0
        self.priority_scale = 1.0
        self.samples_since_stabilize = 0
        self.stabilize_interval = 1000
        
        # Position tracking
        self.position = 0
        self.size = 0

    def _adjust_priority(self, priority, state, reward):
        """Adjust priority based on state uniqueness and reward"""
        state_hash = self._hash_state(state)
        
        # Boost priority for unique states
        if state_hash not in self.state_hashes:
            priority *= self.uniqueness_boost
        else:
            stored_idx, stored_reward = self.state_hashes[state_hash]
            # If this is a higher reward for a similar state, boost priority
            if reward > stored_reward:
                priority *= self.reward_boost_factor

        # Boost priority for high-reward states
        if abs(reward) > 1.0:  # Assuming normalized rewards
            priority *= self.reward_boost_factor * abs(reward)

        return priority

    def push(self, *args):
        with self.lock:
            state, action, reward, next_state, done = args
            
            # Clean up old state hash if needed
            if len(self.state_hashes) >= self.max_hash_entries:
                self.state_hashes.clear()
            
            # Calculate priority with better error handling
            try:
                max_priority = max(self.max_priority, 1.0)
                adjusted_priority = self._adjust_priority(max_priority, state, reward)
            except Exception as e:
                print(f"Priority calculation error: {e}")
                adjusted_priority = max_priority
            
            # Create compressed transition
            compressed_state = CompressedTensor(state, self.compression_level)
            compressed_next_state = CompressedTensor(next_state, self.compression_level)
            transition = Transition(compressed_state, action, reward, 
                                 compressed_next_state, done)
            
            # Handle list insertion/replacement
            if self.size < self.capacity:
                self.memory.append(transition)
                self.size += 1
            else:
                self.memory[self.position] = transition
            
            # Update priority array
            self.priorities[self.position] = adjusted_priority
            
            # Store state hash
            state_hash = self._hash_state(state)
            self.state_hashes[state_hash] = (self.position, reward)
            
            # Update position
            self.position = (self.position + 1) % self.capacity

    def _hash_state(self, state):
        """Create a hash of the state tensor for similarity comparison"""
        if isinstance(state, CompressedTensor):
            state = state.decompress()
        
        # Ensure state is in correct format (N, C, H, W)
        with torch.no_grad():
            if state.dim() == 3:  # (C, H, W)
                state = state.unsqueeze(0)
            
            # Handle stacked frames by taking mean across frame dimension
            if state.shape[1] == 4:  # If we have 4 stacked frames
                state = state.mean(dim=1, keepdim=True)  # Average the frames
                
            # Now state should be (N, 1, H, W)
            state_small = F.interpolate(state, size=(32, 32), mode='bilinear', align_corners=False)
            state_flat = state_small.view(-1)
            state_norm = F.normalize(state_flat, p=2, dim=0)
            return hash(state_norm.cpu().numpy().tobytes())

    def _stabilize_priorities(self):
        """Rescale priorities efficiently"""
        if self.current_size() == 0:
            return
            
        with self.lock:
            # Work with the active portion of priorities
            active_priorities = self.priorities[:self.current_size()]
            
            # Replace NaN/Inf values
            mask = ~np.isfinite(active_priorities)
            if np.any(mask):
                active_priorities[mask] = self.min_priority
            
            # Rescale if needed
            max_pri = np.max(active_priorities)
            if max_pri > 1e6:
                scale = 1.0 / max_pri
                active_priorities *= scale
                self.priority_scale *= scale
            
            self.max_priority = float(np.max(active_priorities))

    def _update_priorities_batch(self):
        """Batch update priorities efficiently"""
        if not self.priority_updates_indices:
            return
            
        indices = np.array(self.priority_updates_indices)
        priorities = np.array(self.priority_updates_values)
        
        # Clean invalid values
        priorities = np.clip(priorities, self.min_priority, None)
        priorities = np.nan_to_num(priorities, nan=self.min_priority)
        
        # Update only valid indices
        valid_mask = indices < self.current_size()
        if np.any(valid_mask):
            self.priorities[indices[valid_mask]] = priorities[valid_mask]
        
        self.priority_updates_indices.clear()
        self.priority_updates_values.clear()

    def current_size(self):
        return self.size

    def sample(self, batch_size):
        with self.lock:
            if self.current_size() == 0:
                return None, None, None
            
            batch_size = min(batch_size, self.current_size())
            
            try:
                # Get active priorities
                priorities = self.priorities[:self.current_size()]
                
                # Safe probability calculation
                probs = np.power(priorities + self.eps, self.alpha)
                probs = probs / np.sum(probs)
                
                # Sample indices and calculate weights
                indices = np.random.choice(self.current_size(), batch_size, p=probs)
                weights = np.power(self.current_size() * probs[indices] + self.eps, -self.beta)
                weights = weights / np.max(weights)
                
                # Get transitions
                samples = [self.memory[idx] for idx in indices]
                
                # Efficient batch decompression
                decompressed_batch = []
                for transition in samples:
                    state = transition.state.decompress()
                    next_state = transition.next_state.decompress()
                    decompressed_batch.append(Transition(
                        state, transition.action, transition.reward,
                        next_state, transition.done
                    ))
                
                weights = torch.tensor(weights, device='cuda', dtype=torch.float32)
                
                # Update beta
                self.beta = min(1.0, self.beta + self.beta_increment)
                
                return decompressed_batch, indices, weights
                
            except Exception as e:
                print(f"Sampling error: {e}")
                return self._fallback_sample(batch_size)

    def _fallback_sample(self, batch_size):
        """Simple uniform sampling as fallback"""
        indices = np.random.choice(self.current_size(), batch_size)
        weights = torch.ones(batch_size, device='cuda', dtype=torch.float32)
        samples = [self.memory[idx] for idx in indices]
        
        decompressed_batch = []
        for transition in samples:
            state = transition.state.decompress()
            next_state = transition.next_state.decompress()
            decompressed_batch.append(Transition(
                state, transition.action, transition.reward,
                next_state, transition.done
            ))
        return decompressed_batch, indices, weights

    def update_priorities(self, indices, priorities):
        """Queue priority updates for batch processing"""
        if torch.is_tensor(priorities):
            priorities = priorities.detach().cpu().numpy()
        
        self.priority_updates_indices.extend(indices)
        self.priority_updates_values.extend(priorities)
        
        if len(self.priority_updates_indices) >= self.update_threshold:
            self._update_priorities_batch()
            self.samples_since_stabilize += self.update_threshold
            
            if self.samples_since_stabilize >= self.stabilize_interval:
                self._stabilize_priorities()
                self.samples_since_stabilize = 0

class EpisodeBuffer:
    def __init__(self, gamma=0.9995, n_step=3):
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
        if (episode_length == 0):
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
    def __init__(self, size=32):
        self.size = size
        # Pre-allocate tensors with correct shapes
        self.cpu_pool = torch.zeros((size, 1, 128, 128),  # Final size only
                                  dtype=torch.uint8,
                                  pin_memory=True)
        self.gpu_pool = torch.zeros((size, 1, 128, 128),
                                  dtype=torch.float16,
                                  device='cuda')
        self.available = set(range(size))
        self.in_use = {}
        self.lock = threading.Lock()
        
        # Create CUDA streams for async operations
        self.streams = [torch.cuda.Stream() for _ in range(2)]

    @torch.no_grad()
    def get_tensors(self):
        with self.lock:
            if not self.available:
                return None, None
                
            idx = self.available.pop()
            cpu_tensor = self.cpu_pool[idx]
            gpu_tensor = self.gpu_pool[idx]
            
            self.in_use[cpu_tensor.data_ptr()] = idx
            return cpu_tensor, gpu_tensor

    def return_tensors(self, cpu_tensor, gpu_tensor):
        with self.lock:
            if cpu_tensor is None or gpu_tensor is None:
                return
                
            ptr = cpu_tensor.data_ptr()
            if ptr in self.in_use:
                idx = self.in_use.pop(ptr)
                self.available.add(idx)
                # Clear the tensors
                self.cpu_pool[idx].zero_()
                self.gpu_pool[idx].zero_()
            # Else silently ignore tensors not from our pool

    def reset(self):
        with self.lock:
            self.available = set(range(self.size))
            self.in_use.clear()
            self.cpu_pool.zero_()
            self.gpu_pool.zero_()
            torch.cuda.empty_cache()

    def __del__(self):
        # Ensure cleanup on deletion
        self.reset()
        del self.cpu_pool
        del self.gpu_pool

def preprocess_state_batch(states, frame_stack=None, preprocess_pool=None):
    """
        Preprocess a batch of states for input to the model.

        Args:
            states: List of states to preprocess
            frame_stack: FrameStack object for stacking frames
            preprocess_pool: PreprocessingPool object for efficient memory management

        Returns:
            Preprocessed tensor of states
    """
    if not isinstance(states, list):
        states = [states]
        
    gpu_tensor = torch.zeros((len(states), 1, 128, 128),
                           dtype=torch.float16, 
                           device='cuda')
    
    with torch.cuda.stream(torch.cuda.current_stream()):
        for i, state in enumerate(states):
            state_tensor = torch.from_numpy(state).float() / 255.0
            if state_tensor.dim() == 2:
                state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)
            elif state_tensor.dim() == 3:
                state_tensor = state_tensor.unsqueeze(0)
                
            resized = F.interpolate(state_tensor, 
                                  size=(128, 128),
                                  mode='nearest')
            
            gpu_tensor[i] = resized.to(dtype=torch.float16)
            
        if frame_stack:
            gpu_tensor = frame_stack(gpu_tensor)
            gpu_tensor = gpu_tensor.reshape(-1, 4, 128, 128)
    
        # Debug output - save SINGLE FRAME, not stacked frames
        # if len(states) > 0:
        #     # Save the original resized frame before stacking
        #     debug_frame = gpu_tensor[0] if not frame_stack else gpu_tensor[0, -1]
            
        #     # Convert to image format - ensure proper scaling
        #     image = debug_frame.cpu().numpy()
        #     if image.max() <= 1.0:  # If normalized
        #         image = (image * 255).clip(0, 255)
        #     image = image.astype(np.uint8)
        #     Image.fromarray(image).save(f'out/{int(time.time())}.png')

    return gpu_tensor.to(memory_format=torch.channels_last)

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

def cleanup(env, display_manager=None, display_enabled=False):
    env.close()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    if display_enabled and display_manager:
        display_manager.stop()

import threading
from queue import Queue
import torch.multiprocessing as mp

class AsyncBufferProcessor:
    def __init__(self, memory, device='cuda', batch_size=32):
        self.memory = memory
        self.device = device
        self.queue = Queue()
        self.processing_thread = threading.Thread(target=self._process_buffer, daemon=True)
        self.processing_thread.start()
        self.current_buffer = EpisodeBuffer(gamma=0.9995)
        self.next_buffer = EpisodeBuffer(gamma=0.9995)
        self.lock = threading.Lock()  # Add thread safety
        self.batch_size = batch_size
        self.batch_buffer = []
        
    def process_batch(self, states, actions, rewards, next_states, dones):
        """Process a batch of transitions in parallel"""
        with torch.cuda.stream(torch.cuda.Stream()):
            # Convert to tensors efficiently
            state_batch = torch.stack(states).pin_memory().to('cuda', non_blocking=True)
            next_state_batch = torch.stack(next_states).pin_memory().to('cuda', non_blocking=True)
            
            self.batch_buffer.extend(zip(
                state_batch.chunk(self.batch_size),
                actions,
                rewards,
                next_state_batch.chunk(self.batch_size),
                dones
            ))
            
            if len(self.batch_buffer) >= self.batch_size:
                self._process_buffer_batch()

    def _process_buffer_batch(self):
        with torch.cuda.stream(torch.cuda.Stream()):
            batch = self.batch_buffer[:self.batch_size]
            self.batch_buffer = self.batch_buffer[self.batch_size:]
            
            states, actions, rewards, next_states, dones = zip(*batch)
            transitions = zip(states, actions, rewards, next_states, dones)
            
            for transition in transitions:
                self.memory.push(*transition)

    def _process_buffer(self):
        while True:
            buffer = self.queue.get()
            if (buffer is None):
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
                self.current_buffer, self.next_buffer = self.next_buffer, EpisodeBuffer(gamma=0.9995)

    def cleanup(self):
        self.queue.put(None)
        self.processing_thread.join()

def soft_update(target_net, policy_net, tau=0.005):
    """Soft update target network parameters"""
    with torch.no_grad():
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * policy_param.data)

class MoveBuffer:
    def __init__(self, max_size=50):
        self.states = []
        self.actions = []
        self.next_states = []
        self.dones = []
        self.max_size = max_size
        
    def push(self, state, action, next_state, done):
        if len(self) >= self.max_size:
            self.clear()  # Prevent buffer from growing too large
            
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.dones.append(done)
        
    def get_last_n(self, n):
        """Get the last n moves from the buffer"""
        n = min(n, len(self))
        return (
            self.states[-n:],
            self.actions[-n:],
            self.next_states[-n:],
            self.dones[-n:]
        )
        
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.next_states.clear()
        self.dones.clear()
        
    def __len__(self):
        return len(self.states)

class GameBuffer(EpisodeBuffer):
    def __init__(self, gamma=0.9995, lookback=10):
        super().__init__(gamma=gamma)
        self.lookback = lookback
        
    def process_game_over(self, reward, move_buffer):
        """Process game over by distributing negative reward across recent moves"""
        # Get last n moves that led to game over
        n = min(self.lookback, len(move_buffer))
        if n == 0:
            return
            
        states, actions, next_states, dones = move_buffer.get_last_n(n)
        reward_per_move = reward / n  # Distribute reward across moves
        
        # Add moves to episode buffer with distributed reward
        for i in range(n):
            # Apply temporal decay - moves further in past get less penalty
            temporal_factor = (i + 1) / n  # Earlier moves get smaller factor
            move_reward = reward_per_move * temporal_factor
            self.push(states[i], actions[i], move_reward, next_states[i], dones[i])
            
    def process_move_sequence(self, move_buffer, reward):
        """Process a sequence of moves that led to a reward"""
        if len(move_buffer) == 0:
            return
            
        # Calculate n-step returns for the sequence
        n = len(move_buffer)
        reward_per_move = reward / n
        
        # Push moves with their portion of the reward
        for i in range(n):
            temporal_factor = (n - i) / n  # Later moves get larger factor
            move_reward = reward_per_move * temporal_factor
            self.push(
                move_buffer.states[i],
                move_buffer.actions[i],
                move_reward,
                move_buffer.next_states[i],
                move_buffer.dones[i]
            )

def main(
    resume=False,
    num_episodes=1000,
    batch_size=128,
    gamma=0.9995,
    patience=10000,
    target_update=5,
    memory_capacity=10000,
    learning_rate=1e-5,
    policy_net_path='tetris_policy_net.pth',
    target_net_path='tetris_target_net.pth',
    max_moves=100,
    save_interval=500,
    weights=None,
    display_enabled=True,
    level=1,
    level_inc=-1,
    cycles=1,
    debug=False
):
    # Define Hyperparameters
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    compute_stream = torch.cuda.Stream()
    memory_stream = torch.cuda.Stream()

    num_actions = 7 # Nothing (removed for now), rotate left / right, move down / left / right, place down, hold

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
        display_manager = DisplayManager()
        display_manager.start()
    else:
        display_manager = None
    
    # Load the weights from json if provided
    if weights:
        if isinstance(weights, str): # Load from file
            with open(weights, 'r') as f:
                weights = json.load(f)
        elif isinstance(weights, dict): # Use directly
            pass


    env = TetrisEnv.TetrisEnv(display_manager=display_manager, max_moves=max_moves, weights=weights, level=level)
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
            max_lr=learning_rate * 30,
            step_size_up=plateau_steps,
            mode='triangular2',
            cycle_momentum=False
        )
    }

    # Temperature scaling for exploration
    temperature = 2.0
    temp_reset = num_episodes // cycles
    temp_decay = 0.995

    # Modified epsilon parameters
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon = epsilon_start
    epsilon_decay_end = 0.8  # Point in cycle where epsilon reaches minimum

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
    
    # Create preprocessing pool with correct shapes
    preprocess_pool = PreprocessingPool(size=32)
    
    # Create CUDA streams for different operations
    compute_stream = torch.cuda.Stream()
    memory_stream = torch.cuda.Stream()
    
    # Create batch processing helper
    batch_processor = AsyncBufferProcessor(memory, device, batch_size=32)
    
    # Add accumulation steps parameter
    accumulation_steps = 4
    
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
                        policy_net.train()
                        target_net.eval()
                        
                        # Initialize move buffer for this episode
                        move_buffer = MoveBuffer(max_size=50)
                        game_buffer = GameBuffer(gamma=gamma, lookback=10)
                        
                        # Episode execution in compute stream
                        with torch.cuda.stream(compute_stream):
                            if episode != 0 and level_inc != -1 and episode % level_inc == 0:
                                env.increase_level()
                            state = env.reset()
                            state = preprocess_state_batch(state, frame_stack, preprocess_pool).to(device)
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

                                with record_function("state_processing") if debug else nullcontext():
                                    with torch.cuda.stream(memory_stream):
                                        next_state = preprocess_state_batch(next_state, frame_stack, preprocess_pool)
                                        next_state = next_state.to(device)
                                        
                                        # Store in move buffer instead of directly to memory
                                        move_buffer.push(state, action, next_state, done)
                                        
                                        # If we got a non-zero reward or the episode is done,
                                        # process the move buffer
                                        if reward != 0 or done:
                                            # Normalize the reward
                                            normalized_reward = reward_normalizer.normalize(
                                                torch.tensor(reward, device=device)
                                            ).item()
                                            
                                            if env.game_state.game_over:
                                                normalized_reward -= 100
                                                # Process game over sequence
                                                game_buffer.process_game_over(normalized_reward, move_buffer)
                                            else:
                                                # Process regular reward sequence
                                                game_buffer.process_move_sequence(move_buffer, normalized_reward)
                                            
                                            # Clear move buffer after processing
                                            move_buffer.clear()
                                            
                                            total_reward += normalized_reward

                                memory_stream.synchronize()
                                state = next_state
                                steps_done += 1

                                # Update exploration parameters
                                # Calculate cycle progress
                                cycle_position = episode % temp_reset
                                cycle_progress = cycle_position / temp_reset

                                # Update temperature (existing code)
                                if cycle_position == 0:
                                    temperature = 2.0
                                else:
                                    progress = cycle_progress
                                    cos_decay = 0.5 * (1 + np.cos(np.pi * progress))
                                    temperature = max(.1, .1 + (2.0 - .1) * cos_decay)

                                # Update epsilon with cyclic decay
                                if cycle_position == 0:
                                    epsilon = epsilon_start
                                else:
                                    if cycle_progress <= epsilon_decay_end:
                                        # Linear decay until 80% of cycle
                                        decay_progress = cycle_progress / epsilon_decay_end
                                        epsilon = epsilon_start + (epsilon_end - epsilon_start) * decay_progress
                                    else:
                                        # Hold at minimum for remainder of cycle
                                        epsilon = epsilon_end

                    # Swap buffers asynchronously
                    buffer_processor.swap_buffers()

                    # Training phase with async prefetch
                    if len(memory) >= batch_size:
                        with torch.cuda.stream(training_stream):
                            # Clear gradients at start of accumulation
                            optimizer.zero_grad(set_to_none=True)
                            
                            # Initialize next_batch before the loop
                            next_batch = memory.sample(batch_size)
                            
                            for batch_idx in range(accumulation_steps):
                                # Get current and prefetch next batch
                                current_batch = next_batch
                                next_batch = memory.sample(batch_size)
                                
                                if current_batch[0] is None:
                                    continue
                                    
                                # Rest of the training loop remains the same
                                batch, indices, weights = current_batch
                                transitions = batch
                                batch = Transition(*zip(*transitions))

                                # Prepare batches efficiently - now with decompressed tensors
                                state_batch = torch.stack(batch.state)
                                action_batch = torch.tensor(batch.action, device=device)
                                reward_batch = torch.tensor(batch.reward, device=device)
                                next_state_batch = torch.stack(batch.next_state)
                                done_batch = torch.tensor(batch.done, device=device, dtype=torch.float16)

                                with autocast(device_type=str(device), dtype=torch.float16):
                                    # Use optimized batch processing
                                    policy_net.train()
                                    current_dist = policy_net.process_batch(state_batch, mode='dqn')
                                    with torch.no_grad():
                                        next_dist = target_net.process_batch(next_state_batch, mode='dqn')

                                    # Get distributions for chosen actions
                                    action_dist = current_dist[range(batch_size), action_batch]
                                    next_q = (next_dist * policy_net.supports).sum(-1)
                                    next_actions = next_q.argmax(1)
                                    next_dist = next_dist[range(batch_size), next_actions]

                                    # Compute categorical projection
                                    proj_dist = compute_categorical_projection(
                                        next_dist, reward_batch, done_batch,
                                        policy_net.supports, gamma,
                                        policy_net.v_min, policy_net.v_max,
                                        policy_net.delta_z, policy_net.num_atoms,
                                        batch_size, device
                                    )

                                    # Calculate loss with gradient accumulation
                                    losses = -(proj_dist * torch.log(action_dist + 1e-8)).sum(-1)
                                    loss = (weights * losses).mean() / accumulation_steps
                                    
                                    # Scale loss and accumulate gradients
                                    scaler.scale(loss).backward()

                                    # Update priorities based on TD error
                                    td_error = torch.abs(losses).detach()
                                    memory.update_priorities(indices, td_error)

                                    episode_losses.append(loss.item() * accumulation_steps)

                                # Step optimizer after accumulation
                                if ((batch_idx + 1) % accumulation_steps == 0):
                                    scaler.unscale_(optimizer)
                                    clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
                                    scaler.step(optimizer)
                                    scaler.update()
                                    optimizer.zero_grad(set_to_none=True)

                            # Use soft updates for target network
                            soft_update(target_net, policy_net)

                        # Synchronize streams before next iteration
                        torch.cuda.synchronize()

                        # Explicit cleanup after each training step
                        del state_batch, action_batch, reward_batch, next_state_batch, done_batch
                        torch.cuda.empty_cache()

                    policy_net.eval()
                    # Periodic memory cleanup
                    if episode % 5 == 0:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()

                    # Update networks and learning rate - moved after training
                    optimizer.step()

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
                        policy_net.reset_noise()
                        target_net.reset_noise()


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
        cleanup(env, display_manager, display_enabled)
    finally:
        buffer_processor.cleanup()
        # Ensure thorough cleanup
        cleanup(env, display_manager, display_enabled)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return best_reward

def compute_categorical_projection(next_dist, reward_batch, done_batch, supports, gamma, v_min, v_max, delta_z, num_atoms, batch_size, device):
    """Helper function to compute categorical projection"""
    with torch.cuda.stream(torch.cuda.current_stream()):
        # Ensure input shapes are correct and on GPU
        reward_batch = reward_batch.unsqueeze(-1).to(device)  # [B, 1]
        done_batch = done_batch.unsqueeze(-1).to(device)     # [B, 1]
        supports = supports.unsqueeze(0).to(device)          # [1, A] where A is num_atoms
        
        # Compute projected atoms [B, A]
        projected_atoms = reward_batch + (1 - done_batch) * gamma * supports
        projected_atoms = torch.clamp(projected_atoms, v_min, v_max)
        
        # Get indices for projection [B, A]
        bj = (projected_atoms - v_min) / delta_z
        l = bj.floor().long()
        u = bj.ceil().long()
        
        # Handle corner cases maintaining shapes
        l = torch.where((u > 0) * (l == u), l - 1, l)
        u = torch.where((l < (num_atoms - 1)) * (l == u), u + 1, u)
        
        # Create properly shaped offset tensor [B, A]
        offset = torch.arange(0, batch_size, device=device).unsqueeze(1)
        offset = offset * num_atoms  # Scale offset by num_atoms
        
        # Add offset to indices while maintaining shapes
        l = l + offset  # [B, A]
        u = u + offset  # [B, A]
        
        # Compute weights [B, A]
        u_weights = (u.float() - bj).reshape(-1)
        l_weights = (bj - l.float()).reshape(-1)
        
        # Prepare flattened tensors
        l_idx = l.reshape(-1)  # [B*A]
        u_idx = u.reshape(-1)  # [B*A]
        next_dist_flat = next_dist.reshape(-1)  # [B*A]
        
        # Initialize output distribution
        proj_dist = torch.zeros_like(next_dist, device=device)
        proj_dist_flat = proj_dist.reshape(-1)
        
        # Perform scatter operations with validated shapes
        proj_dist_flat.scatter_add_(0, l_idx, next_dist_flat * u_weights)
        proj_dist_flat.scatter_add_(0, u_idx, next_dist_flat * l_weights)
        
        # Reshape to original batch shape [B, A]
        return proj_dist.reshape(batch_size, -1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DDQN on Tetris')
    parser.add_argument('--resume', type=bool, default=None)
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--gamma', type=float, default=0.9995)
    parser.add_argument('--target_update', type=int, default=5)
    parser.add_argument('--memory_capacity', type=int, default=10000)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--policy_net', type=str, default='tetris_policy_net.pth')
    parser.add_argument('--target_net', type=str, default='tetris_target_net.pth')
    parser.add_argument('--max_moves', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--level', type=int, default=1)
    parser.add_argument('--level_inc', type=int, default=-1)
    parser.add_argument('--cycles', type=int, default=1)
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
        level=args.level,
        level_inc=args.level_inc,
        cycles=args.cycles,
        debug=args.debug,
        display_enabled=False
    )