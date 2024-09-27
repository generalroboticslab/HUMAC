from Dataloader.Hide_and_Seek_Dataset_real_IL_direction import HideandSeekDataset
import os
from torch.utils.data import ConcatDataset
from torch.utils.data import  DataLoader
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from Model.Resnet_PE_H import ResNet18_PE_H,train
import json

green = '\033[92m'
red = '\033[91m'
blue = '\033[94m'
yellow = '\033[93m'
reset = '\033[0m'

seed_value = 42
torch.manual_seed(seed_value)
np.random.seed(seed_value)
import random
random.seed(seed_value)

root_folder = "data"

num_of_frames = 3
step_ahead = 15

for folder in os.scandir(root_folder):
    num_traj = 0
    total_fail = 0
    total_episode = 0
    total_time = 0

    base_folder = os.path.join(root_folder, folder.name)
    num_seekers = int(str(folder.name)[0])

    # if num_seekers != 3:
    #     continue
    
    
    for seed_num in os.scandir(base_folder):
        num_traj += 1
        # print(base_folder,seed_num.name)
        for agent_id in range(0,num_seekers):
            dataset = HideandSeekDataset(os.path.join(base_folder,seed_num.name),agent_id,num_seekers,num_of_frame=num_of_frames,step_ahead=step_ahead)
            
            if (dataset.have_missing_agent()):
                print("BAD")
            
            if num_traj <= 45:
                try:
                    all_training_data = ConcatDataset([all_training_data,dataset])
                except:
                    all_training_data = dataset
            else:
                try:
                    all_validation_data = ConcatDataset([all_validation_data,dataset])
                except:
                    all_validation_data = dataset
            
        total_episode += 1
        total_fail += int(dataset.is_seeker_fail())
        total_time += int(len(dataset))
 

    print(f"{folder.name}, Total Episode: {total_episode}, Seeker Success Rate:{100 - total_fail/total_episode*100:.2f}%, Average Time: {total_time/5/total_episode:.2f}s")
    # break
print(f"Total Training Data Amount: {len(all_training_data)}")
print(f"Total Validation Data Amount: {len(all_validation_data)}")

model = ResNet18_PE_H(num_of_frames)
criterion = nn.MSELoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)


#load Encoder 1
original_first_layer_weights = model.resnet18_1.conv1.weight.data.cuda()

state_dict = torch.load("Weights/IL_Long_split_by_traj_3frames_15step_ahead_direction/model_epoch41.pth")["model_state_dict"]
updated_state_dict = {}
for key in state_dict.keys():
    if 'fc' not in key:
        new_key = key.replace('resnet18.', 'resnet18_1.')
        updated_state_dict[new_key] = state_dict[key]

model.load_state_dict(updated_state_dict,strict=False)

first_layer_weights = model.resnet18_1.conv1.weight.data.cuda()
loaded_first_layer_weights = state_dict['resnet18.conv1.weight'].cuda()

if torch.equal(first_layer_weights, loaded_first_layer_weights) and not torch.equal(loaded_first_layer_weights,original_first_layer_weights):
    print("Encoder1 Weights loaded successfully!")
else:
    print("Error: Encoder1 Weights not loaded correctly.")

#load Encoder2
original_first_layer_weights = model.resnet18_2.conv1.weight.data.cuda()

state_dict = torch.load("Weights/Team_Encoder_3frames_15step_ahead/model_epoch53.pth")["model_state_dict"]
updated_state_dict = {}
for key in state_dict.keys():
    if 'fc' not in key:
        new_key = key.replace('resnet18.', 'resnet18_2.')
        updated_state_dict[new_key] = state_dict[key]

model.load_state_dict(updated_state_dict,strict=False)

first_layer_weights = model.resnet18_2.conv1.weight.data.cuda()
loaded_first_layer_weights = state_dict['resnet18.conv1.weight'].cuda()

if torch.equal(first_layer_weights, loaded_first_layer_weights) and not torch.equal(loaded_first_layer_weights,original_first_layer_weights):
    print("Encoder2 Weights loaded successfully!")
else:
    print("Error: Encoder2 Weights not loaded correctly.")

#freeze the weight of Encoder 1 and 2
for param in model.resnet18_1.parameters():
    param.requires_grad = False
print(f"{green}Encoder 1 Freeze{reset}")

for param in model.resnet18_2.parameters():
    param.requires_grad = False
print(f"{green}Encoder 2 Freeze{reset}")

batch_size = 512

trainloader = DataLoader(all_training_data , batch_size=batch_size, shuffle=True,num_workers=35)
valloader = DataLoader(all_validation_data , batch_size=batch_size, shuffle=True,num_workers=35)

print(f"\nTraining data size: {len(all_training_data)}")
print(f"Validation data size: {len(all_validation_data)}\n")

save_base_directory = f"Weights/IL_PE-H_{num_of_frames}frames_{step_ahead}step_ahead_direction"

import sys
sys.setrecursionlimit(10000000)

learning_rate = 0.001
dataset_name = base_folder.split("-1")[-1]
optimizer_name = "Adam"
loss_name = "MSE"

t_l,v_l = train(model,
          50,
          'cuda',
          trainloader,valloader,
          criterion,
          optimizer,
          save_base_directory,
          learning_rate,
          dataset_name,
          batch_size,
          optimizer_name,
          loss_name,
)

training_result = {"Training_loss":t_l,"Validation_loss":v_l}
with open(save_base_directory+'/training_result.json', 'w') as f:
    json.dump(training_result, f)





