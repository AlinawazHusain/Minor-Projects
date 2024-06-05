import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Define the max pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Define the transpose convolutional layers for upsampling
        self.trans_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2)
        self.trans_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.trans_conv3 = nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2)
        
    def forward(self, x):
        # Apply convolutional layers with ReLU activation and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Apply transpose convolutional layers for upsampling
        x = F.relu(self.trans_conv1(x))
        x = F.relu(self.trans_conv2(x))
        x = F.relu(self.trans_conv3(x))
        
        return x

# Initialize the model
# model = Model()

# x = torch.randn(1,3,460,460)
# print(model(x).shape)