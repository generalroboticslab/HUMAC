from torch import nn
from torchvision.models import resnet18
import torch
import torch.nn.functional as F

class CustomResNet18(nn.Module):
    def __init__(self, num_of_frames,num_seekers=1):
        super(CustomResNet18, self).__init__()

        self.resnet18 = resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(4 * num_of_frames, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.resnet18.fc = nn.Identity()
        
        self.fc1 = nn.Linear(512, 256) 
        self.fc2 = nn.Linear(256, 128)  
        self.fc3 = nn.Linear(128, 2*num_seekers) 

    def forward(self, x):
        with torch.no_grad():
            latent = self.resnet18(x)
            latent= F.relu(self.fc1(latent))
            latent  = F.relu(self.fc2(latent))
            y = self.fc3(latent)
            return y
        

class ResNet18_PE_H(nn.Module):
    def __init__(self, num_of_frames):
        super(ResNet18_PE_H, self).__init__()

        self.resnet18_1 = resnet18(pretrained=False)
        self.resnet18_1.conv1 = nn.Conv2d(4 * num_of_frames, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18_1.fc = nn.Identity()

        self.resnet18_2 = resnet18(pretrained=False)
        self.resnet18_2.conv1 = nn.Conv2d(4 * num_of_frames, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18_2.fc = nn.Identity()
        
        self.fc1 = nn.Linear(512*2, 256) 
        self.fc2 = nn.Linear(256, 128)    
        self.fc3 = nn.Linear(128, 2)      

    def forward(self, x):

        latent1 = self.resnet18_1(x)
        latent2 = self.resnet18_2(x)

        latent = torch.cat((latent1,latent2),dim=1)
        
        latent = F.relu(self.fc1(latent))
        latent = F.relu(self.fc2(latent))
        y = self.fc3(latent)
        return y

