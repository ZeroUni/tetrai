import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017, device='cuda'):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features).to(device))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features).to(device))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features).to(device))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features).to(device))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features).to(device))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features).to(device))
        
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / (self.in_features ** 0.5)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init)
    
    def reset_noise(self):
        epsilon_in = self._f(torch.randn(self.in_features, device=self.device))
        epsilon_out = self._f(torch.randn(self.out_features, device=self.device))
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._f(torch.randn(self.out_features, device=self.device)))
    
    def _f(self, x):
        return x.sign().mul_(x.abs().sqrt())
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class TetrisCNN(nn.Module):
    def __init__(self, num_actions: int, device='cuda', sigma_init=0.5): 
        super(TetrisCNN, self).__init__()
        print("Using device: ", device)
        
        # Simplified architecture with proper size calculations
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2).to(device)  # Output: 32x32x32
        self.bn1 = nn.BatchNorm2d(32).to(device)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1).to(device)  # Output: 64x16x16
        self.bn2 = nn.BatchNorm2d(64).to(device)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1).to(device)  # Output: 128x16x16
        self.bn3 = nn.BatchNorm2d(128).to(device)

        # All you need is attention
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4).to(device)
        
        # Calculate flattened size
        self.flat_size = 128 * 16 * 16
        
        self.fc1 = NoisyLinear(self.flat_size, 512, device=device, sigma_init=sigma_init)
        self.fc2 = NoisyLinear(512, num_actions, device=device, sigma_init=sigma_init)

    def forward(self, x):
        # Ensure contiguous memory layout
        x = x.contiguous()
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Attention layer
        batch_size = x.size(0)
        x_att = x.view(batch_size, 128, -1).permute(2, 0, 1)

        x_out, _ = self.attention(x_att, x_att, x_att)

        x = x_out.permute(1, 2, 0).view(batch_size, 128, 16, 16)

        x = x + x_att.permute(1, 2, 0).view(batch_size, 128, 16, 16)
        
        # Use reshape instead of view
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
    def reset_noise(self):
        for layer in self.children():
            if hasattr(layer, 'reset_noise'):
                layer.reset_noise()

class RainbowTetrisCNN(nn.Module):
    def __init__(
        self,
        num_actions: int,
        device='cuda',
        num_atoms=51,
        sigma_init=0.5,
        v_min=-10,
        v_max=10
    ):
        super(RainbowTetrisCNN, self).__init__()
        print("Using device: ", device)
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.device = device
        self.v_min = v_min
        self.v_max = v_max

        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        ).to(device)

        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4).to(device)

        feature_size = 128 * 16 * 16

        # Dueling DQN Heads
        self.dqn_advantage = nn.Sequential(
            NoisyLinear(feature_size, 512, device=device, sigma_init=sigma_init),
            nn.ReLU(),
            NoisyLinear(512, num_actions * num_atoms, device=device, sigma_init=sigma_init)
        )

        self.dqn_value = nn.Sequential(
            NoisyLinear(feature_size, 512, device=device, sigma_init=sigma_init),
            nn.ReLU(),
            NoisyLinear(512, num_atoms, device=device, sigma_init=sigma_init)
        )

        # PPO Heads for further optimization
        self.ppo_actor = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
            nn.Softmax(dim=-1)
        )

        self.ppo_critic = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

        # Register buffers for the atoms
        self.register_buffer('supports', torch.linspace(v_min, v_max, num_atoms).to(device))
        self.register_buffer('delta_z', torch.tensor((v_max - v_min) / (num_atoms - 1)).to(device))

        # Convert model to channels_last memory format for better GPU utilization
        self = self.to(memory_format=torch.channels_last)

    def extract_features(self, x):
        # Convert input to channels_last for better performance
        x = x.contiguous(memory_format=torch.channels_last)
        x = self.features(x)

        batch_size = x.size(0)
        # More efficient reshape using flatten
        x_att = x.flatten(2).transpose(1, 2)
        
        # Optimize attention computation
        x_out, _ = self.attention(x_att, x_att, x_att)
        
        # Efficient reshape back to original format
        x = x_out.transpose(1, 2).view(batch_size, 128, 16, 16)
        x = x.add_(x_att.transpose(1, 2).view(batch_size, 128, 16, 16))
        
        return x.flatten(1)

    def dqn_forward(self, x):
        x = self.extract_features(x)
        advantage = self.dqn_advantage(x).reshape(-1, self.num_actions, self.num_atoms)
        value = self.dqn_value(x).reshape(-1, 1, self.num_atoms)
        # Expand value to match advantage shape for proper broadcasting
        value = value.expand(-1, self.num_actions, self.num_atoms)
        # Calculate Q-values using dueling architecture
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + advantage - advantage_mean
        return F.softmax(q_values, dim=-1)

    def ppo_forward(self, x):
        x = self.extract_features(x)
        # Compute both outputs in parallel
        return self.ppo_actor(x), self.ppo_critic(x)

    def get_action(self, x, mode='dqn'):
        with torch.no_grad(), autocast():
            if mode == 'dqn':
                probs = self.dqn_forward(x)
                q_values = torch.sum(probs * self.supports, dim=-1)
                return torch.argmax(q_values, dim=-1)
            elif mode == 'ppo':
                action_probs, _ = self.ppo_forward(x)
                return torch.distributions.Categorical(action_probs).sample()

    def forward(self, x, mode='dqn'):
        if mode == 'dqn':
            return self.dqn_forward(x)
        elif mode == 'ppo':
            return self.ppo_forward(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
    def reset_noise(self):
        for layer in self.children():
            if hasattr(layer, 'reset_noise'):
                layer.reset_noise()

# Actions are: rotate right, rotate left, push down / left / right, force down, hold, NOTHING - 8 actions