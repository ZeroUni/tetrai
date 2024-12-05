import torch.optim as optim
import torch.nn.functional as F
import torch

import torchvision.transforms as transforms

from collections import namedtuple, deque

import numpy as np

from PIL import Image

import random

from tetrisCNN import TetrisCNN
from tetrisCNN import TetrisEnv

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


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
    
def preprocess_state(state):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return preprocess(state).unsqueeze(0)

def main():
    # Define Hyperparameters
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
    our_device = torch.device("cuda")
    policy_net = TetrisCNN(num_actions).to(our_device)
    target_net = TetrisCNN(num_actions).to(our_device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Get our tetris ENV (TODO)
    env = TetrisEnv.TetrisEnv()

    optimizer = optim.Adam(policy_net.parameters())
    memory = ReplayMemory(memory_capacity)

    steps_done = 0

    # Start training?
    for episode in range(num_episodes):
        state = env.reset()
        state = preprocess_state(state).to(our_device)
        total_reward = 0
        done = False

        while not done:
            # What
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
            np.exp(-1. * steps_done / epsilon_decay)

            if random.random() < epsilon:
                action = random.randrange(num_actions)
            else:
                with torch.no_grad():
                    action = policy_net(state).argmax(dim = 1).item()

            # Do the thing
            next_state, reward, done = env.step(action)
            next_state = preprocess_state(next_state).to(our_device)
            total_reward += reward

            # Remember the thing
            memory.push(state, action, reward, next_state, done)
            state = next_state
            steps_done += 1

            # Sample a batch and optimize model
            if len(memory) > batch_size:
                transitions = memory.sample(batch_size)
                batch = Transition(*zip(*transitions))

                # Prepare batches
                state_batch = torch.cat(batch.state)
                action_batch = torch.tensor(batch.action, device=our_device).unsqueeze(1)
                reward_batch = torch.tensor(batch.reward, device=our_device)
                next_state_batch = torch.cat(batch.next_state)
                done_batch = torch.tensor(batch.done, device=our_device, dtype=torch.float)

                # Compute Q(s_t, a)
                q_values = policy_net(state_batch).gather(1, action_batch)

                # Compute target Q values
                with torch.no_grad():
                    next_state_actions = policy_net(next_state_batch).argmax(1, keepdim=True)
                    next_q_values = target_net(next_state_batch).gather(1, next_state_actions).squeeze()
                    expected_q_values = reward_batch + (gamma * next_q_values * (1 - done_batch))

                # Compute loss
                loss = F.mse_loss(q_values.squeeze(), expected_q_values)

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update the target network
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode}: Total Reward = {total_reward}")

if __name__ == "__main__":
    main()