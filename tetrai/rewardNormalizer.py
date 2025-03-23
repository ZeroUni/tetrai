import torch

class RewardNormalizer:
    def __init__(self, clip_limit=10.0, decay=0.99, epsilon=1e-8):
        self.running_mean = torch.zeros(1, device='cuda')
        self.running_var = torch.ones(1, device='cuda')
        self.clip_limit = clip_limit
        self.decay = decay
        self.epsilon = epsilon

    def record(self, reward):
        """Store reward without normalizing it"""
        self.episode_rewards.append(reward)
        return reward  # Return raw reward for now

    @torch.no_grad() 
    def normalize(self, reward):
        """Normalize all rewards in the episode using consistent statistics"""
        if not self.episode_rewards:
            return []
            
        # Convert stored rewards to tensor
        rewards = torch.tensor(self.episode_rewards, device='cuda', dtype=torch.float32)
        
        # Update running statistics based on entire episode
        episode_mean = rewards.mean()
        episode_std = rewards.std() + self.epsilon
        
        # Update running statistics with entire episode (optional)
        self.running_mean = self.decay * self.running_mean + (1 - self.decay) * episode_mean
        self.running_var = self.decay * self.running_var + (1 - self.decay) * (rewards.var() + self.epsilon)
        
        # Normalize using episode statistics
        normalized = (rewards - episode_mean) / episode_std
        
        # Apply soft clipping
        normalized = torch.tanh(normalized / self.clip_limit) * self.clip_limit
        
        return normalized.tolist()

    def reset(self):
        """Reset statistics"""
        self.running_mean.zero_()
        self.running_var.fill_(1.)