import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time
from torchvision.models import resnet18
import torch.nn.functional as F

class IL(nn.Module):
    def __init__(self,num_of_frames,decision_frequency,num_of_seekers,max_teammate_num = 1):
        super(IL, self).__init__()

        self.resnet18 = resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(4 * num_of_frames, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc = nn.Identity()
        
        self.fc1 = nn.Linear(512, 256) 
        self.fc2 = nn.Linear(256, 128)  
        self.fc3 = nn.Linear(128, 2*max_teammate_num)   

        self.resnet18.eval()

        self.replay_buffer = []
        for _ in range(num_of_seekers):
            self.replay_buffer.append([])
        
        self.num_of_frames = num_of_frames
        self.num_of_frames_per_second= int(1/decision_frequency)
        self.num_of_seekers = num_of_seekers
        self.max_teammate_num = max_teammate_num
        
        self.last_id = []
        for _ in range(num_of_seekers):
            self.last_id.append(0)
    
    def forward(self, *input_tensor):

        with torch.inference_mode():
            x,id = input_tensor
            if id.shape[0] != 1:
                id = id[:,2].squeeze()
            else:
                id = id[:,2]
            x = torch.cat([obs for obs in x],dim = 0)
            
            if torch.is_tensor(x) and x.shape[0] == 4*self.num_of_seekers:

                output = torch.zeros((self.num_of_seekers,2*self.max_teammate_num)).cuda() ## number of seekers
                for i in range(self.num_of_seekers):
                    obs1 = x[4*i:4*(i+1),:,:].unsqueeze(0)
                    #update replay buffer 
                    if obs1.shape[1] == 4:
                        # reset replay if new episode
                        if id[i] > self.last_id[i]:
                            self.replay_buffer[i] = []
                        
                        self.last_id[i] = id[i]
                        
                        if len(self.replay_buffer[i]) < (self.num_of_frames-1)*self.num_of_frames_per_second+1: 
                            self.replay_buffer[i].append(obs1)
                        else:
                            self.replay_buffer[i].pop(0)
                            self.replay_buffer[i].append(obs1)
                        
                        #stack observation

                        obs_list = []
                        ind_list = []
                        for m in range(self.num_of_frames):
                            if len(self.replay_buffer[i]) > self.num_of_frames_per_second*m and len(self.replay_buffer[i]) <= self.num_of_frames_per_second*(m+1):
                                for f in range(m):
                                    obs_list.append(self.replay_buffer[i][-1-f*self.num_of_frames_per_second])
                                    ind_list.append(-1-f*self.num_of_frames_per_second)
                                for _ in range(self.num_of_frames-m):
                                    obs_list.append(self.replay_buffer[i][-1-m*self.num_of_frames_per_second])
                                    ind_list.append(-1-m*self.num_of_frames_per_second)
                                break
                        obs_list.reverse()
                        ind_list.reverse()
                        obs1 = torch.cat(obs_list,1)
                        

                    if np.shape(obs1)[1] == 4*self.num_of_frames: 

                        with torch.no_grad():
                            obs1 = self.resnet18(obs1)
                            # Forward pass through the new FC layers
                            obs1 = F.relu(self.fc1(obs1))
                            obs1 = F.relu(self.fc2(obs1))
                            output[i,:] = self.fc3(obs1)

                    else:
                        pass
            else:
                output = torch.zeros((self.num_of_seekers,2*self.max_teammate_num)).cuda()
                
        return output


class PE_H(nn.Module):
    def __init__(self,num_of_frames,decision_frequency,num_of_seekers):
        super(PE_H, self).__init__()
        # Load a pre-trained ResNet18 model
        self.resnet18_1 = resnet18(pretrained=False)
        self.resnet18_1.conv1 = nn.Conv2d(4 * num_of_frames, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18_1.fc = nn.Identity()

        self.resnet18_2 = resnet18(pretrained=False)# Replace the first convolution layer
        self.resnet18_2.conv1 = nn.Conv2d(4 * num_of_frames, 64, kernel_size=7, stride=2, padding=3, bias=False) # Remove the original fully connected layer
        self.resnet18_2.fc = nn.Identity()
        
        # Add three new fully connected layers
        self.fc1 = nn.Linear(512*2, 256)  # First FC layer
        self.fc2 = nn.Linear(256, 128)        # Second FC layer
        self.fc3 = nn.Linear(128, 2)          # Third FC layer, output 3 classes

        #turn the model into evaluation mode
        self.resnet18_1.eval()
        self.resnet18_2.eval()
        

        self.replay_buffer = []
        for _ in range(num_of_seekers): #number of seekers
            self.replay_buffer.append([])
        
        self.num_of_frames = num_of_frames
        self.num_of_frames_per_second= int(1/decision_frequency)
        self.num_of_seekers = num_of_seekers

        self.last_id = []
        for _ in range(num_of_seekers): #number of seekers
            self.last_id.append(0)
    
    def forward(self, *input_tensor):
        with torch.inference_mode():
            x,id = input_tensor
            if id.shape[0] != 1:
                id = id[:,2].squeeze()
            else:
                id = id[:,2]
            x = torch.cat([obs for obs in x],dim = 0)
            
            if torch.is_tensor(x) and x.shape[0] == 4*self.num_of_seekers:
                # print(x.shape)

                output = torch.zeros((self.num_of_seekers,2)).cuda() ## number of seekers
                for i in range(self.num_of_seekers):
                    obs1 = x[4*i:4*(i+1),:,:].unsqueeze(0)
                    # print(obs1.shape)

                    #update replay buffer 
                    if obs1.shape[1] == 4:
                        # reset replay if new episode
                        if id[i] > self.last_id[i]:
                            self.replay_buffer[i] = []
                        
                        self.last_id[i] = id[i]
                        # print(len(self.replay_buffer[0]),len(self.replay_buffer[1]))
                        if len(self.replay_buffer[i]) < (self.num_of_frames-1)*self.num_of_frames_per_second+1: 
                            self.replay_buffer[i].append(obs1)
                        else:
                            self.replay_buffer[i].pop(0)
                            self.replay_buffer[i].append(obs1)
                        
                        #stack observation

                        obs_list = []
                        ind_list = []
                        for m in range(self.num_of_frames):
                            if len(self.replay_buffer[i]) > self.num_of_frames_per_second*m and len(self.replay_buffer[i]) <= self.num_of_frames_per_second*(m+1):
                                for f in range(m):
                                    obs_list.append(self.replay_buffer[i][-1-f*self.num_of_frames_per_second])
                                    ind_list.append(-1-f*self.num_of_frames_per_second)
                                for _ in range(self.num_of_frames-m):
                                    obs_list.append(self.replay_buffer[i][-1-m*self.num_of_frames_per_second])
                                    ind_list.append(-1-m*self.num_of_frames_per_second)
                                break
                        obs_list.reverse()
                        ind_list.reverse()
                        print(ind_list)
                        obs1 = torch.cat(obs_list,1)
                        

                    if np.shape(obs1)[1] == 4*self.num_of_frames: 

                        with torch.no_grad():
                            latent1 = self.resnet18_1(obs1)
                            latent2 = self.resnet18_2(obs1)

                            obs1 = torch.cat((latent1,latent2),dim=1)
                            # Forward pass through the new FC layers
                            obs1 = F.relu(self.fc1(obs1))
                            obs1 = F.relu(self.fc2(obs1))
                            output[i,:] = self.fc3(obs1)

                    else:
                        pass
            else:
                output = torch.zeros((self.num_of_seekers,2)).cuda()
                
        return output


class PE_N(nn.Module):
    def __init__(self,num_of_frames,decision_frequency,num_of_seekers):
        super(PE_N, self).__init__()
        # Load a pre-trained ResNet18 model
        self.teammate_prediction_model = IL(num_of_frames,decision_frequency,num_of_seekers,max_teammate_num = 3)

        self.resnet18 = resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(4 * num_of_frames, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc = nn.Identity()
        
        # Add three new fully connected layers
        self.fc1 = nn.Linear(512, 128)  # First FC layer
        self.fc2 = nn.Linear(256, 128)        # Second FC layer
        self.fc3 = nn.Linear(128, 2)          # Third FC layer, output 3 classes
        self.fc4 = nn.Linear(3*2, 128)          # Third FC layer, output 3 classes

        #turn the model into evaluation mode
        self.resnet18.eval()
        self.teammate_prediction_model.eval()
        

        self.replay_buffer = []
        for _ in range(num_of_seekers): #number of seekers
            self.replay_buffer.append([])
        
        self.num_of_frames = num_of_frames
        self.num_of_frames_per_second= int(1/decision_frequency)
        self.num_of_seekers = num_of_seekers

        self.last_id = []
        for _ in range(num_of_seekers): #number of seekers
            self.last_id.append(0)
    
    def forward(self, *input_tensor):
        with torch.inference_mode():
            x,id = input_tensor
            y = self.teammate_prediction_model(*input_tensor)
            if id.shape[0] != 1:
                id = id[:,2].squeeze()
            else:
                id = id[:,2]
            x = torch.cat([obs for obs in x],dim = 0)
            
            if torch.is_tensor(x) and x.shape[0] == 4*self.num_of_seekers:
                # print(x.shape)

                output = torch.zeros((self.num_of_seekers,2)).cuda() ## number of seekers
                for i in range(self.num_of_seekers):
                    obs1 = x[4*i:4*(i+1),:,:].unsqueeze(0)
                    
                    obs2 = y[i,:]

                    #update replay buffer 
                    if obs1.shape[1] == 4:
                        # reset replay if new episode
                        if id[i] > self.last_id[i]:
                            self.replay_buffer[i] = []
                        
                        self.last_id[i] = id[i]
                        if len(self.replay_buffer[i]) < (self.num_of_frames-1)*self.num_of_frames_per_second+1: 
                            self.replay_buffer[i].append(obs1)
                        else:
                            self.replay_buffer[i].pop(0)
                            self.replay_buffer[i].append(obs1)
                        
                        #stack observation

                        obs_list = []
                        ind_list = []
                        for m in range(self.num_of_frames):
                            if len(self.replay_buffer[i]) > self.num_of_frames_per_second*m and len(self.replay_buffer[i]) <= self.num_of_frames_per_second*(m+1):
                                for f in range(m):
                                    obs_list.append(self.replay_buffer[i][-1-f*self.num_of_frames_per_second])
                                    ind_list.append(-1-f*self.num_of_frames_per_second)
                                for _ in range(self.num_of_frames-m):
                                    obs_list.append(self.replay_buffer[i][-1-m*self.num_of_frames_per_second])
                                    ind_list.append(-1-m*self.num_of_frames_per_second)
                                break
                        obs_list.reverse()
                        ind_list.reverse()
    
                        obs1 = torch.cat(obs_list,1)
                       
                        

                    if np.shape(obs1)[1] == 4*self.num_of_frames: 

                        with torch.no_grad():
                            latent = self.resnet18(obs1)
                            obs1 = F.relu(torch.cat((self.fc1(latent),self.fc4(obs2).unsqueeze(0)),dim=1))
                            obs1 = F.relu(self.fc2(obs1))
                            output[i,:] = self.fc3(obs1)

                    else:
                        pass
            else:
                output = torch.zeros((self.num_of_seekers,2)).cuda()
                
        return output

class FT(IL):
    def __init__(self,num_of_frames,decision_frequency,num_of_seekers):
        super(FT, self).__init__(num_of_frames,decision_frequency,num_of_seekers)

    def forward(self, *input_tensor):
        return super(FT, self).forward(*input_tensor)


class FT_Team(IL):
    def __init__(self,num_of_frames,decision_frequency,num_of_seekers):
        super(FT_Team, self).__init__(num_of_frames,decision_frequency,num_of_seekers)

    def forward(self, *input_tensor):
        return super(FT_Team, self).forward(*input_tensor)

class Heuristic(nn.Module):
    def __init__(self,num_of_frames,decision_frequency,num_of_seekers):
        super(Heuristic, self).__init__()
        self.num_of_frames = num_of_frames
        self.num_of_frames_per_second= int(1/decision_frequency)
        self.num_of_seekers = num_of_seekers

    def forward(self, *input_tensor):
        output = torch.zeros((self.num_of_seekers,2)).cuda() + 51.0
        return output

class Mix(nn.Module):
    def __init__(self,base_policy,addon_policy,num_of_frames,decision_frequency,num_of_seekers):
        super(Mix, self).__init__()
        self.base_policy = eval(base_policy)(num_of_frames,decision_frequency,num_of_seekers)
        self.addon_policy = eval(addon_policy)(num_of_frames,decision_frequency,num_of_seekers)
        self.num_of_frames = num_of_frames
        self.num_of_frames_per_second= int(1/decision_frequency)
        self.num_of_seekers = num_of_seekers

    def forward(self, *input_tensor):
        x,id = input_tensor
        output = torch.cat((self.base_policy(x,id),self.addon_policy(x,id)),dim=1)
        return output