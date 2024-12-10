import torch

class RewardNormalizer:
    def __init__(self, clip_limit=10.0, decay=0.99, epsilon=1e-8):
        self.running_mean = torch.zeros(1, device='cuda')
        self.running_var = torch.ones(1, device='cuda')
        self.clip_limit = clip_limit
        self.decay = decay
        self.epsilon = epsilon

    @torch.no_grad() 
    def normalize(self, reward):
        """Normalize reward while preserving sign and relative magnitude"""
        # Update running statistics
        self.running_mean = self.decay * self.running_mean + (1 - self.decay) * reward
        
        # Update running variance 
        diff = reward - self.running_mean
        self.running_var = self.decay * self.running_var + (1 - self.decay) * (diff ** 2)
        
        # Normalize using running statistics while preserving sign
        std = torch.sqrt(self.running_var + self.epsilon)
        normalized = diff / std
        
        # Apply soft clipping to preserve relative magnitudes
        normalized = torch.tanh(normalized / self.clip_limit) * self.clip_limit
        
        return normalized

    def reset(self):
        """Reset statistics"""
        self.running_mean.zero_()
        self.running_var.fill_(1.)