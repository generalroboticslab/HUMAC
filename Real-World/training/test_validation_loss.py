from Dataloader.Hide_and_Seek_Dataset_real_IL import HideandSeekDataset
import os
from torch.utils.data import ConcatDataset
from torch.utils.data import  DataLoader
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from Model.Imitation_learning import CNN, train
from Model.Resnet import CustomResNet18
import json
from torch import nn
from torchvision.models import resnet18
from matplotlib import pyplot as plt
import time 

root_folder = "data"

num_of_frames = 3
step_ahead = 5
ns = 3

def plot(observation,action,num_of_frame=3):
    fig, axs = plt.subplots(2, num_of_frame,figsize=(20,10))
      
    img_height, img_width = (2.66*2,2.5*2)

    for i in range(1,num_of_frame+1):
    # Calculate the extent of the image to center it at (0, 0)
        extent = [-img_width / 2, img_width / 2, -img_height / 2, img_height / 2]

        # Display the image with the specified extent
        axs[0,i-1].imshow(observation[(i-1)*4:i*4-1,:,:].permute(1,2,0), extent=extent)
        
        # axs[0, i - 1].invert_xaxis()
        # axs[0, i - 1].invert_yaxis()

        if (i == num_of_frame):
            axs[0,i-1].scatter(action[0], action[1], c='orange', marker='+', s=40)
        # axs[0,i-1].scatter(-position[0], -position[1], c='blue', marker='o', s=30)
        
        # print(position)
        # print(teammate_position[0,teammate_index],teammate_position[1,teammate_index])
        
        
        
        binary_mask = observation[4*i-1,:,:]
        # img_height, img_width = (57.98,58.18)
        extent = [-img_width / 2, img_width / 2, -img_height / 2, img_height / 2]
        axs[1,i-1].imshow(binary_mask, extent=extent, cmap='gray')
        
        # axs[1, i - 1].invert_xaxis()
        # axs[1, i - 1].invert_yaxis()
    plt.savefig(f"turn_plot/{time.time()}.png",dpi=500)

for folder in os.scandir(root_folder):
    num_traj = 0
    total_fail = 0
    total_episode = 0
    total_time = 0

    base_folder = os.path.join(root_folder, folder.name)
    num_seekers = int(str(folder.name)[0])

    # if num_seekers != ns:
    #     continue
    
    
    for seed_num in os.scandir(base_folder):
        num_traj += 1
        # print(base_folder,seed_num.name)
        for agent_id in range(0,num_seekers):
            dataset = HideandSeekDataset(os.path.join(base_folder,seed_num.name),agent_id,num_seekers,num_of_frame=num_of_frames,step_ahead=step_ahead)
            
            if (dataset.have_missing_agent()):
                print("BAD")
            
            if num_traj <= 45:
            #     try:
            #         all_training_data = ConcatDataset([all_training_data,dataset])
            #     except:
            #         all_training_data = dataset
            # else:
                try:
                    all_validation_data = ConcatDataset([all_validation_data,dataset])
                except:
                    all_validation_data = dataset
            
        total_episode += 1
        total_fail += int(dataset.is_seeker_fail())
        total_time += int(len(dataset))
 

    print(f"{folder.name}, Total Episode: {total_episode}, Seeker Success Rate:{100 - total_fail/total_episode*100:.2f}%, Average Time: {total_time/5/total_episode:.2f}s")
    # break
print(f"Total Validation Data Amount: {len(all_validation_data)}")


model = resnet18(weights=False)
model.conv1 = nn.Conv2d(4*num_of_frames,64,kernel_size=7,stride =2, padding=3, bias=False)
model.fc = nn.Linear(512,2)
model = CustomResNet18(num_of_frames)

weight = torch.load("Weights/IL_Long_split_by_traj_3frames_5step_ahead/model_epoch65.pth")
model.load_state_dict(weight["model_state_dict"])
model.eval()


batch_size = 512

filtered_obs_list = []
filtered_action_list = []


valloader = DataLoader(all_validation_data , batch_size=batch_size, shuffle=True, num_workers=35)

count = 0
for obs, action,_,t in valloader:
    for i in range(len(obs)):
        if action[i,-1] == 1:
            filtered_obs_list.append(obs[i].unsqueeze(0))
            filtered_action_list.append(action[i,:-1].unsqueeze(0))
            # break
    # break


# Convert the lists to a new dataset
filtered_obs = torch.cat(filtered_obs_list)
filtered_actions = torch.cat(filtered_action_list)

# print(f"Filtered data size: {len(filtered_obs)}")
# print(f"Filtered data size: {len(filtered_actions)}")
# Create a new ConcatDataset with the filtered data
filtered_dataset = ConcatDataset([(filtered_obs[i], filtered_actions[i]) for i in range(len(filtered_obs))])
from torch.utils.data import random_split

# Calculate lengths for training and validation splits
total_len = len(filtered_dataset)
train_len = int(0.95 * total_len)
val_len = total_len - train_len

# Split the dataset
train_dataset, val_dataset = random_split(filtered_dataset, [train_len, val_len])

# Create DataLoaders for both datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=35)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=35)

print(f"Training Data Amount: {len(train_dataset)}")
print(f"Validation Data Amount: {len(val_dataset)}")

model.eval()           
    
total_loss = 0
for obs, action in val_loader:
    with torch.no_grad():
        output = model(obs)
        loss = nn.MSELoss()
        total_loss += loss(output, action)
print(f"Validation Loss: {total_loss/len(val_loader)}")



# print(f"\nTraining data size: {len(train_data)}")
# print(f"Validation data size: {len(val_data)}\n")
