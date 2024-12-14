import torch
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import numpy as np
import argparse
import os
import time
import json
import threading
import multiprocessing
from collections import deque
import traceback
import random

import TetrisCNN
import TetrisEnv
from rewardNormalizer import RewardNormalizer
from DDQNonCNN import display_images, preprocess_state_batch, FrameStack, PreprocessingPool

class PPOMemory:
    def __init__(self, batch_size=32):
        self.states = []
        self.actions = []
        self.probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
        self.advantages = None
        self.returns = None

    def store(self, state, action, prob, val, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.values.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return batches

    def get_batch(self, batch_indices):
        states = torch.stack([self.states[i] for i in batch_indices])
        actions = torch.tensor([self.actions[i] for i in batch_indices], dtype=torch.long)
        probs = torch.tensor([self.probs[i] for i in batch_indices])
        values = torch.tensor([self.values[i] for i in batch_indices])
        rewards = torch.tensor([self.rewards[i] for i in batch_indices])
        dones = torch.tensor([self.dones[i] for i in batch_indices])
        return states, actions, probs, values, rewards, dones

    def compute_advantages(self, gamma, gae_lambda, next_value):
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device='cuda')
        values = torch.tensor(self.values, dtype=torch.float32, device='cuda')
        dones = torch.tensor(self.dones, dtype=torch.float32, device='cuda')
        
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
            
        self.advantages = advantages
        self.returns = advantages + values
        
    def get_batches(self):
        n_states = len(self.states)
        batch_starts = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states)
        np.random.shuffle(indices)
        
        for start in batch_starts:
            batch_idx = indices[start:start + self.batch_size]
            yield (
                torch.stack([self.states[i] for i in batch_idx]),
                torch.tensor([self.actions[i] for i in batch_idx], device='cuda'),
                torch.tensor([self.probs[i] for i in batch_idx], device='cuda'),
                self.advantages[batch_idx],
                self.returns[batch_idx]
            )

def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    advantages = []
    gae = 0
    
    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[step + 1]
            
        delta = rewards[step] + gamma * next_val * (1 - dones[step]) - values[step]
        gae = delta + gamma * gae_lambda * (1 - dones[step]) * gae
        advantages.insert(0, gae)
        
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + torch.tensor(values)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages, returns

def main(
    num_episodes=1000,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    policy_clip=0.2,
    value_clip=0.2,
    n_epochs=4,
    batch_size=32,
    max_moves=100,
    save_interval=100,
    model_path='ppo_model.pth',
    weights=None,
    display_enabled=True,
    level=1,
    level_inc=-1,
    debug=False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    
    # Create model directory
    model_dir = f'out/ppo_{int(time.time())}'
    os.makedirs(model_dir, exist_ok=True)

    # Load the weights from json if provided
    if weights:
        if isinstance(weights, str): # Load from file
            with open(weights, 'r') as f:
                weights = json.load(f)
        elif isinstance(weights, dict): # Use directly
            pass

    # Initialize model
    model = TetrisCNN.RainbowTetrisCNN(
        num_actions=8,
        device=device
    ).to(device)

    # Initialize optimizer with AdamW
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )

    # Initialize PPO memory
    reward_normalizer = RewardNormalizer()
    scaler = GradScaler()
    frame_stack = FrameStack(stack_size=4)
    preprocess_pool = PreprocessingPool(size=32)

    # Setup display if enabled
    if display_enabled:
        render_queue = multiprocessing.Queue()
        display_thr = threading.Thread(target=display_images, args=(render_queue,))
        display_thr.start()
    else:
        display_thr = None
        render_queue = None

    # Initialize environment
    env = TetrisEnv.TetrisEnv(render_queue, max_moves=max_moves, weights=weights, level=level)
    
    # Training tracking
    best_reward = float('-inf')
    results = {
        'episode_rewards': [],
        'policy_losses': [],
        'value_losses': [],
        'learning_rates': []
    }

    try:
        for episode in range(num_episodes):
            memory = PPOMemory(batch_size)
            state = env.reset()
            state = preprocess_state_batch(state, frame_stack, preprocess_pool).to(device)
            done = False
            total_reward = 0

            # Collect trajectory
            while not done:
                with torch.no_grad(), autocast(device_type='cuda'):
                    # Get action probabilities and value
                    action_probs, state_value = model(state, mode='ppo')
                    
                    # Apply action masking
                    action_mask = env.get_legal_actions_mask().to(device)
                    
                    # Use a larger epsilon for numerical stability
                    epsilon = 1e-8
                    
                    # Clamp probabilities to prevent extremely small values
                    action_probs = torch.clamp(action_probs, epsilon, 1.0)
                    
                    # Apply mask and handle zero probabilities safely
                    masked_probs = action_probs * action_mask
                    mask_sum = masked_probs.sum(dim=-1, keepdim=True)
                    
                    # Safe division with fallback
                    masked_probs = torch.where(
                        mask_sum > epsilon,
                        masked_probs / (mask_sum + epsilon),
                        action_mask.float() / (action_mask.sum() + epsilon)
                    )
                    
                    # Additional safety check
                    if torch.isnan(masked_probs).any() or torch.isinf(masked_probs).any():
                        # Fallback to uniform distribution over legal actions
                        masked_probs = action_mask.float() / (action_mask.sum() + epsilon)
                    
                    # Verify and normalize probabilities
                    masked_probs = F.normalize(masked_probs, p=1, dim=-1)
                    
                    # Create distribution with safety checks
                    try:
                        dist = Categorical(probs=masked_probs.squeeze(0))
                        action = dist.sample()
                        action_prob = dist.log_prob(action)
                    except Exception as e:
                        print(f"Distribution error: {e}")
                        # Fallback to random legal action
                        legal_actions = torch.where(action_mask.squeeze(0) == 1)[0]
                        action = legal_actions[torch.randint(0, len(legal_actions), (1,))]
                        action_prob = torch.log(torch.tensor(1.0 / len(legal_actions)))

                # Execute action
                next_state, reward, done = env.step(action.item())
                reward = reward_normalizer.normalize(torch.tensor(reward, device=device)).item()
                next_state = preprocess_state_batch(next_state, frame_stack, preprocess_pool).to(device)

                # Store transition
                memory.store(
                    state.detach(),
                    action.item(),
                    action_prob.item(),
                    state_value.item(),
                    reward,
                    done
                )

                state = next_state
                total_reward += reward

            # End of episode updates
            with torch.no_grad(), autocast(device_type='cuda'):
                _, final_value = model(state, mode='ppo')
                memory.compute_advantages(gamma, gae_lambda, final_value.item())

            # PPO update
            for _ in range(n_epochs):
                # In the PPO update section:
                for states, actions, old_probs, advantages, returns in memory.get_batches():
                    with autocast(device_type='cuda'):
                        new_probs, values = model(states, mode='ppo')
                        
                        # Safe probability handling
                        epsilon = 1e-8
                        new_probs = torch.clamp(new_probs, epsilon, 1.0)
                        
                        # Apply action masking
                        action_masks = torch.stack([
                            env.get_legal_actions_mask() for _ in range(len(states))
                        ]).to(device)
                        
                        # Safe masked probabilities calculation
                        masked_probs = new_probs * action_masks
                        mask_sums = masked_probs.sum(dim=-1, keepdim=True)
                        
                        # Handle zero probabilities safely
                        masked_probs = torch.where(
                            mask_sums > epsilon,
                            masked_probs / (mask_sums + epsilon),
                            action_masks.float() / (action_masks.sum(dim=-1, keepdim=True) + epsilon)
                        )
                        
                        # Normalize and verify
                        masked_probs = F.normalize(masked_probs, p=1, dim=-1)
                        
                        # Safe ratio calculation
                        dist = Categorical(probs=masked_probs)
                        new_action_probs = dist.log_prob(actions)
                        ratio = torch.exp(torch.clamp(new_action_probs - old_probs, -20, 20))
                        
                        # Clamp ratios for stability
                        ratio = torch.clamp(ratio, 0.1, 10.0) 
                        
                        surr1 = ratio * advantages
                        surr2 = torch.clamp(ratio, 1-policy_clip, 1+policy_clip) * advantages
                        policy_loss = -torch.min(surr1, surr2).mean()

                        # Value loss
                        value_pred = values  # Already squeezed in forward pass
                        value_loss = F.mse_loss(value_pred, returns)
                        
                        loss = policy_loss + 0.5 * value_loss
                    
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)

                    # Add stricter gradient clipping
                    clip_grad_norm_(model.parameters(), max_norm=0.5)

                    # Safe optimizer step
                    scaler.step(optimizer)
                    scaler.update()

                    results['policy_losses'].append(policy_loss.item())
                    results['value_losses'].append(value_loss.item())

            # Clear memory after updates
            memory.clear()

            # Track results
            results['episode_rewards'].append(total_reward)
            results['learning_rates'].append(optimizer.param_groups[0]['lr'])

            print(f"Episode {episode}: Total Reward = {total_reward}")
            reward_normalizer.reset()

            # Save model periodically
            if episode % save_interval == 0:
                torch.save(model.state_dict(), f'{model_dir}/model_{episode}.pth')
                with open(f'{model_dir}/results.json', 'w') as f:
                    json.dump(results, f)

            # Increment level if specified
            if level_inc != -1 and episode % level_inc == 0:
                env.increase_level()

    except Exception as e:
        print(f"Training error: {e}")
        traceback.print_exc()
    finally:
        # Final cleanup
        if display_enabled:
            render_queue.put(None)
            display_thr.join()
        env.close()
        torch.cuda.empty_cache()

    # Save final model and results
    torch.save(model.state_dict(), f'{model_dir}/model_final.pth')
    with open(f'{model_dir}/results.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO on Tetris')
    parser.add_argument('--num_episodes', type=int, default=2000, help='Number of episodes to train')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for PPO')
    parser.add_argument('--max_moves', type=int, default=100, help='Maximum number of moves per episode')
    parser.add_argument('--save_interval', type=int, default=200, help='Interval to save model')
    parser.add_argument('--weights', type=str, default=None, help='Path to weights file')
    parser.add_argument('--level', type=int, default=1, help='Starting level')
    parser.add_argument('--level_inc', type=int, default=-1, help='Level increment interval')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    main(
        num_episodes=args.num_episodes,
        learning_rate=args.learning_rate,
        max_moves=args.max_moves,
        save_interval=args.save_interval,
        weights=args.weights,
        level=args.level,
        level_inc=args.level_inc,
        debug=args.debug
    )