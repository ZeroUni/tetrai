import torch
from torch.amp import autocast, GradScaler
import torch.optim as optim
import torch.nn.functional as F

import torchvision.transforms as transforms

from collections import namedtuple, deque

import numpy as np

from PIL import Image

import threading
import multiprocessing
import cv2

import pygame

import random

import TetrisCNN

import TetrisEnv

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
        # Stack frames along channel dimension
        stacked = torch.cat(list(self.frames), dim=0)
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
    
def preprocess_state(state, frame_stack=None):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.RandomRotation(10),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    processed = preprocess(state)
    if frame_stack:
        return frame_stack(processed)
    return processed

def render_thread(env):
    while True:
        env.render()
        pygame.time.wait(env.render_delay)

def display_images(render_queue):
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

def main():
    # Define Hyperparameters
    try:
        num_actions = 9 # Nothing, rotate left / right, move up / down / left / right, place down, hold
        batch_size = 32
        gamma = 0.99
        epsilon_start = 1.0
        epsilon_decay = 5000
        epsilon_end = 0.1
        target_update = 10
        memory_capacity = 10000
        num_episodes = 1000
        learning_rate = 1e-4

        # Initialize our networks and torch optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {device}')
        
        policy_net = TetrisCNN.TetrisCNN(num_actions).to(device)
        target_net = TetrisCNN.TetrisCNN(num_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        # Use a scalar for our gradients

        # Get our tetris ENV (TODO)
        render_queue = multiprocessing.Queue()
        env = TetrisEnv.TetrisEnv(render_queue)
        env.render_mode = False

        display_proc = multiprocessing.Process(target=display_images, args=(render_queue,))
        display_proc.start()

        optimizer = optim.Adam(policy_net.parameters())
        memory = ReplayMemory(memory_capacity)
        frame_stack = FrameStack(stack_size=4)
        scaler = GradScaler(init_scale=2.**16, growth_factor=2., backoff_factor=0.5, growth_interval=1000, enabled=True)

        steps_done = 0

        # Start training?
        for episode in range(num_episodes):
            state = env.reset()
            state = preprocess_state(state, frame_stack).to(device)
            total_reward = 0
            done = False

            while not done:
                # Select action using epsilon-greedy policy
                epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                        np.exp(-1. * steps_done / epsilon_decay)

                if random.random() < epsilon:
                    action = random.randrange(num_actions)
                else:
                    with torch.no_grad():
                        action = policy_net(state.unsqueeze(0)).argmax(dim=1).item()

                # Perform action
                next_state, reward, done = env.step(action)
                next_state = preprocess_state(next_state, frame_stack).to(device)
                total_reward += reward

                # Store transition in memory
                memory.push(state, action, reward, next_state, done)
                state = next_state
                steps_done += 1

                # Sample a batch and optimize model
                if len(memory) >= batch_size:
                    transitions = memory.sample(batch_size)
                    batch = Transition(*zip(*transitions))

                    # Prepare batches
                    state_batch = torch.stack(batch.state)
                    action_batch = torch.tensor(batch.action, device=device).unsqueeze(1)
                    reward_batch = torch.tensor(batch.reward, device=device)
                    next_state_batch = torch.stack(batch.next_state)
                    done_batch = torch.tensor(batch.done, device=device, dtype=torch.float)

                    # Double DQN: Select action using policy_net, evaluate using target_net
                    with autocast(device_type=device.__str__()):
                        # Actions selected by policy_net
                        next_actions = policy_net(next_state_batch).argmax(dim=1, keepdim=True)
                        # Q-values from target_net for the selected actions
                        next_q_values = target_net(next_state_batch).gather(1, next_actions).squeeze()
                        expected_q_values = reward_batch + gamma * next_q_values * (1 - done_batch)

                        # Compute current Q-values
                        q_values = policy_net(state_batch).gather(1, action_batch).squeeze()

                        # Compute loss
                        loss = F.mse_loss(q_values, expected_q_values.detach())

                    # Optimize the model
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            # Update the target network
            if episode % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            print(f"Episode {episode}: Total Reward = {total_reward}")


        env.close()
        render_queue.put(None)
        display_proc.join()

        torch.save(policy_net.state_dict(), 'tetris_policy_net.pth')
        torch.save(target_net.state_dict(), 'tetris_target_net.pth')
    except Exception as e:
        print(e)
        env.close()
        display_proc.terminate()


if __name__ == "__main__":
    main()