import torch.nn as nn
import torch.nn.functional as F

class TetrisCNN(nn.Module):
    def __init__(self, num_actions: int): 
        super(TetrisCNN, self).__init__()
        # Set up some convolution layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(128)
        # Set up connected layer
        self.fc1 = nn.Linear(128 * 5 * 5, 512)
        
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