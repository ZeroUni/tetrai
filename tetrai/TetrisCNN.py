import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
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
        epsilon_in = self._f(torch.randn(self.in_features))
        epsilon_out = self._f(torch.randn(self.out_features))
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._f(torch.randn(self.out_features)))
    
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
    def __init__(self, num_actions: int): 
        super(TetrisCNN, self).__init__()
        # Set up some convolution layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)  # Changed from 1 to 4
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(128)
        # Set up connected layer
        self.fc1 = nn.Linear(128 * 10 * 10, 512)
        
        # Modify our Q values for each output
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        # Shape our input
        x = F.relu(self.bn1(self.conv1(x)))  
        x = F.relu(self.bn2(self.conv2(x)))  
        x = F.relu(self.bn3(self.conv3(x)))  
        x = F.relu(self.bn4(self.conv4(x)))  
        # Flatten our input
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        
        # Get our returned values from our batch
        x = self.fc2(x)
        return x

# Actions are: rotate right, rotate left, push up / down / left / right, force down, hold, NOTHING - 9 actions