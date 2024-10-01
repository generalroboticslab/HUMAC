from torch import nn
from torchvision.models import resnet18
import torch.nn.functional as F
import torch 
import os
import time
import wandb
from itertools import zip_longest

class IL(nn.Module):
    def __init__(self, num_of_frames,num_seekers=1):
        super(IL, self).__init__()
        self.resnet18 = resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(4 * num_of_frames, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc = nn.Identity()
        
        self.fc1 = nn.Linear(512, 256)  # First FC layer
        self.fc2 = nn.Linear(256, 128)        # Second FC layer
        self.fc3 = nn.Linear(128, 2*num_seekers)          # Third FC layer, output 3 classes

    def forward(self, x):
        x = self.resnet18(x)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PE_N(nn.Module):
    def __init__(self, num_of_frames,max_num_teammate=3):
        super(PE_N, self).__init__()
        self.resnet18 = resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(4 * num_of_frames, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc = nn.Identity()
        
        self.fc1 = nn.Linear(512, 128)  # First FC layer
        self.fc2 = nn.Linear(256, 128)        # Second FC layer
        self.fc3 = nn.Linear(128, 2)          # Third FC layer, output 3 classes
        
        self.fc4 = nn.Linear(max_num_teammate*2, 128)

        self.teammate_prediction = IL(num_of_frames,num_seekers=max_num_teammate)
        

    def forward(self, x):
        with torch.no_grad():
            y = self.teammate_prediction(x)
        x = self.resnet18(x)
        x = F.relu(torch.cat((self.fc1(x),self.fc4(y)),dim=1))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class PE_H(nn.Module):
    def __init__(self, num_of_frames):
        super(PE_H, self).__init__()
        # Load a pre-trained ResNet18 model
        self.resnet18_1 = resnet18(pretrained=False)
        self.resnet18_1.conv1 = nn.Conv2d(4 * num_of_frames, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18_1.fc = nn.Identity()

        self.resnet18_2 = resnet18(pretrained=False)# Replace the first convolution layer
        self.resnet18_2.conv1 = nn.Conv2d(4 * num_of_frames, 64, kernel_size=7, stride=2, padding=3, bias=False) # Remove the original fully connected layer
        self.resnet18_2.fc = nn.Identity()
        
        self.fc1 = nn.Linear(512*2, 256)  # First FC layer
        self.fc2 = nn.Linear(256, 128)        # Second FC layer
        self.fc3 = nn.Linear(128, 2)          # Third FC layer, output 3 classes

    def forward(self, x):
        latent1 = self.resnet18_1(x)
        latent2 = self.resnet18_2(x)

        x = torch.cat((latent1,latent2),dim=1)
        
        # Forward pass through the new FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# def PE_T(IL):
#     def __init__(self,num_of_frames,num_seekers):
#         super(PE_T, self).__init__(num_of_frames,num_seekers = num_seekers)

#     def forward(self, *input_tensor):
#         return super(PE_T, self).forward(*input_tensor)