import torch
from torch.amp import autocast, GradScaler
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
    return processed.float()

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

def main():
    # Define Hyperparameters
    try:
        parser = argparse.ArgumentParser(description='Train DDQN on Tetris')
        parser.add_argument('--resume', type=bool, default=None)
        parser.add_argument('--num_episodes', type=int, default=1000)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--target_update', type=int, default=10)
        parser.add_argument('--memory_capacity', type=int, default=10000)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--policy_net', type=str, default='tetris_policy_net.pth')
        parser.add_argument('--target_net', type=str, default='tetris_target_net.pth')
        parser.add_argument('--max_moves', type=int, default=-1)
        parser.add_argument('--save_interval', type=int, default=500)

        args = parser.parse_args()


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
        
        policy_net = TetrisCNN.TetrisCNN(num_actions).to(device)
        target_net = TetrisCNN.TetrisCNN(num_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        # Use a scalar for our gradients

        # Get our tetris ENV (TODO)
        render_queue = multiprocessing.Queue()
        env = TetrisEnv.TetrisEnv(render_queue, max_moves=args.max_moves)
        env.render_mode = False

        display_proc = multiprocessing.Process(target=display_images, args=(render_queue,))
        display_proc.start()

        optimizer = optim.Adam(policy_net.parameters())
        memory = ReplayMemory(memory_capacity)
        frame_stack = FrameStack(stack_size=4)
        scaler = GradScaler()

        steps_done = 0

        if args.resume:
            policy_net.load_state_dict(torch.load(args.policy_net))
            target_net.load_state_dict(torch.load(args.target_net))

        # Start training?
        for episode in range(num_episodes):
            state = env.reset()
            state = preprocess_state(state, frame_stack).to(device)
            total_reward = 0
            done = False

            while not done:
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
                    state_batch = torch.stack(batch.state).to(device)
                    action_batch = torch.tensor(batch.action, device=device).unsqueeze(1)
                    reward_batch = torch.tensor(batch.reward, device=device)
                    next_state_batch = torch.stack(batch.next_state).to(device)
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
                        loss = F.mse_loss(q_values.float(), expected_q_values.detach().float())

                    # Optimize the model
                    scaler.scale(loss).to(device).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    # Reset the noise in the noisy layers
                    policy_net.reset_noise()
                    target_net.reset_noise()




            # Update the target network
            if episode % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            print(f"Episode {episode}: Total Reward = {total_reward}")

            if episode != 0 and episode % args.save_interval == 0:
                torch.save(policy_net.state_dict(), f'{model_dir}/tetris_policy_net_{episode}.pth')
                torch.save(target_net.state_dict(), f'{model_dir}/tetris_target_net_{episode}.pth')


        env.close()
        render_queue.put(None)
        display_proc.join()

        torch.save(policy_net.state_dict(), f'{model_dir}/tetris_policy_net_final.pth')
        torch.save(target_net.state_dict(), f'{model_dir}/tetris_target_net_final.pth')
    except Exception as e:
        print(e)
        traceback.print_exc()
        env.close()
        display_proc.terminate()
        display_proc.join()


if __name__ == "__main__":
    main()