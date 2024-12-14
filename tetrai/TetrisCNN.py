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

        self.register_buffer('noise_scale', torch.full((1,), 0.5))
        self.noise_scale_decay = 0.99995  # Slower decay for long-term learning

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
            # Scale noise over time
            self.noise_scale.mul_(self.noise_scale_decay)
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon * self.noise_scale
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon * self.noise_scale
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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        identity = self.skip(x)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        x = self.bn2(self.conv2(x))
        x = F.leaky_relu(x + identity, 0.1)
        return x
    
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

        # Input: [B, 4, 128, 128]
        self.features = nn.Sequential(
            # [B, 4, 128, 128] -> [B, 64, 32, 32]
            nn.Conv2d(4, 64, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.1),

            # [B, 64, 32, 32] -> [B, 128, 32, 32]
            ResidualBlock(64, 128),
            # [B, 128, 32, 32] -> [B, 128, 16, 16]
            nn.MaxPool2d(2),

            # [B, 128, 16, 16] -> [B, 128, 16, 16]
            ResidualBlock(128, 128),
            # [B, 128, 16, 16] -> [B, 128, 8, 8]
            nn.MaxPool2d(2),

            # [B, 128, 8, 8] -> [B, 128, 8, 8]
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
        ).to(device)

        # Attention with correct embedding dimension
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4).to(device)

        # Final feature size after convolutions and pooling
        feature_size = 128 * 8 * 8  # Corrected size

        # Dueling DQN Heads
        self.dqn_advantage = nn.Sequential(
            NoisyLinear(feature_size, 1024, device=device, sigma_init=sigma_init),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(1024),
            NoisyLinear(1024, 512, device=device, sigma_init=sigma_init),
            nn.LeakyReLU(0.1),
            NoisyLinear(512, num_actions * num_atoms, device=device, sigma_init=sigma_init)
        )

        self.dqn_value = nn.Sequential(
            NoisyLinear(feature_size, 1024, device=device, sigma_init=sigma_init),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(1024),
            NoisyLinear(1024, 512, device=device, sigma_init=sigma_init),
            nn.LeakyReLU(0.1),
            NoisyLinear(512, num_atoms, device=device, sigma_init=sigma_init)
        )

        # Updated PPO Actor head with similar structure to DQN
        self.ppo_actor = nn.Sequential(
            NoisyLinear(feature_size, 1024, device=device, sigma_init=sigma_init),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(1024),
            NoisyLinear(1024, 512, device=device, sigma_init=sigma_init),
            nn.LeakyReLU(0.1),
            NoisyLinear(512, num_actions, device=device, sigma_init=sigma_init)
        ).to(device)

        # Updated PPO Critic head with similar structure to DQN
        self.ppo_critic = nn.Sequential(
            NoisyLinear(feature_size, 1024, device=device, sigma_init=sigma_init),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(1024),
            NoisyLinear(1024, 512, device=device, sigma_init=sigma_init),
            nn.LeakyReLU(0.1),
            NoisyLinear(512, 1, device=device, sigma_init=sigma_init)
        ).to(device)

        # Register buffers for the atoms
        self.register_buffer('supports', torch.linspace(v_min, v_max, num_atoms).to(device))
        self.register_buffer('delta_z', torch.tensor((v_max - v_min) / (num_atoms - 1)).to(device))

        # Convert model to channels_last memory format for better GPU utilization
        self = self.to(memory_format=torch.channels_last)

    def extract_features(self, x):
        # Ensure input has correct shape for channels_last format
        if len(x.shape) != 4:
            x = x.reshape(-1, 4, 128, 128)
        
        # Forward through convolutional layers
        # Input: [B, 4, 128, 128] -> Output: [B, 128, 8, 8]
        x = self.features(x)
        
        batch_size = x.size(0)
        channels = x.size(1)  # 128
        height = x.size(2)    # 8
        width = x.size(3)     # 8
        seq_len = height * width  # 64
        
        # Reshape for attention
        # [B, C, H, W] -> [B, C, H*W] -> [H*W, B, C]
        x_att = x.view(batch_size, channels, seq_len)
        x_att = x_att.permute(2, 0, 1)  # [64, B, 128]
        
        # Apply attention
        x_out, _ = self.attention(x_att, x_att, x_att)
        
        # Reshape back
        # [64, B, 128] -> [B, 128, 8, 8]
        x = x_out.permute(1, 2, 0).view(batch_size, channels, height, width)
        
        # Residual connection
        x_residual = x_att.permute(1, 2, 0).view(batch_size, channels, height, width)
        x = x + x_residual
        
        # Final flatten for FC layers
        return x.reshape(batch_size, -1)

    def dqn_forward(self, x):
        x = self.extract_features(x)
        advantage = self.dqn_advantage(x)
        value = self.dqn_value(x)
        
        # Reshape advantage to [batch, actions, atoms]
        advantage = advantage.view(-1, self.num_actions, self.num_atoms)
        # Reshape value to [batch, 1, atoms]
        value = value.view(-1, 1, self.num_atoms)
        
        # Proper advantage normalization
        advantage = advantage - advantage.mean(dim=1, keepdim=True)
        
        # Combine value and advantage
        q_dist = value + advantage
        return F.softmax(q_dist, dim=-1)
    
    def ppo_forward(self, x):
        x = self.extract_features(x)
        
        # Apply actor network and get action distributions
        action_logits = self.ppo_actor(x)
        log_probs = F.log_softmax(action_logits, dim=-1)
        action_probs = log_probs.exp()
        
        # Apply critic network
        state_value = self.ppo_critic(x)
        
        return action_probs, log_probs, state_value.squeeze(-1)
        
    def process_batch(self, states, mode='dqn'):
        """Efficiently process a batch of states in parallel"""
        with torch.cuda.stream(torch.cuda.current_stream()):
            with autocast(device_type='cuda', dtype=torch.float16):
                # Ensure correct input shape [B, 4, 128, 128]
                if states.shape[-2:] != (128, 128):
                    raise ValueError(f"Expected input shape [..., 128, 128], got {states.shape}")
                
                features = self.extract_features(states)
                
                if mode == 'dqn':
                    # Process DQN outputs maintaining precision for value estimation
                    advantage = self.dqn_advantage(features).reshape(-1, self.num_actions, self.num_atoms)
                    value = self.dqn_value(features).reshape(-1, 1, self.num_atoms)
                    value = value.expand(-1, self.num_actions, -1)
                    
                    # Calculate Q-values using dueling architecture
                    advantage_mean = advantage.mean(dim=1, keepdim=True)
                    q_dist = value + (advantage - advantage_mean)
                    
                    # Apply softmax and ensure gradients
                    q_dist = F.softmax(q_dist, dim=-1)
                    
                    return q_dist
                    
                elif mode == 'ppo':
                    # Process PPO outputs with improved precision handling
                    action_logits = self.ppo_actor(features)
                    log_probs = F.log_softmax(action_logits, dim=-1)
                    action_probs = log_probs.exp()
                    state_values = self.ppo_critic(features)
                    
                    return action_probs, log_probs, state_values.squeeze(-1)

    def get_action(self, x, mode='dqn'):
        with torch.no_grad(), autocast():
            # Ensure input shape
            if x.dim() == 3:
                x = x.unsqueeze(0)  # Add batch dimension
            
            if mode == 'dqn':
                probs = self.dqn_forward(x)
                q_values = torch.sum(probs * self.supports, dim=-1)
                return torch.argmax(q_values, dim=-1)
            elif mode == 'ppo':
                action_probs, _, _ = self.ppo_forward(x)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                return action, dist.log_prob(action)
            
    def forward(self, x, mode='dqn'):
        # Use optimized batch processing
        return self.process_batch(x, mode)
        
    def reset_noise(self):
        for layer in self.children():
            if hasattr(layer, 'reset_noise'):
                layer.reset_noise()

# Actions are: rotate right, rotate left, push down / left / right, force down, hold, NOTHING - 8 actions