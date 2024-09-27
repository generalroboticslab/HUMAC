from Dataloader.Hide_and_Seek_Dataset_Human import HideandSeekDataset
import os
from torch.utils.data import ConcatDataset
from torch.utils.data import  DataLoader
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from Model.Imitation_learning import CNN, train
from Model.Resnet import CustomResNet18, fine_tune
import json
from torch import nn
from torchvision.models import resnet18

seed_value = 42
torch.manual_seed(seed_value)
np.random.seed(seed_value)
import random
random.seed(seed_value)

root_folder = "Dataset_human_click_short"

total_time = 0
num_of_frames=5

for folder in os.scandir(root_folder):
    
    total_fail = 0
    total_episode = 0
    total_frames = 0


    base_folder = os.path.join(root_folder, folder.name)
    num_seekers = int(str(folder.name)[0])

    for seed_num in os.scandir(base_folder):
        for agent_id in range(num_seekers):
            dataset = HideandSeekDataset(os.path.join(base_folder,seed_num.name),num_seekers, agent_id, human_control=True,collect_decision_freqeuncy=0.2,train_decision_frequency=1)
            
            total_fail += int(dataset.is_seeker_fail())
            
            if (dataset.have_missing_agent()):
                print("BAD")
            
            try:
                all_data = ConcatDataset([all_data,dataset])
            except:
                all_data = dataset
            
            dataset1 = HideandSeekDataset(os.path.join(base_folder,seed_num.name),num_seekers, agent_id, human_control=False,collect_decision_freqeuncy=0.2,train_decision_frequency=1)
            
            total_fail += int(dataset1.is_seeker_fail())
            
            if (dataset1.have_missing_agent()):
                print("BAD")
            
            try:
                all_data1 = ConcatDataset([all_data1,dataset1])
            except:
                all_data1 = dataset1


        total_episode += 1

        total_frames += dataset.total_len()

        # if total_episode == 5:
        #     break

    print(f"{folder.name}, Total Episode: {total_episode}, Seeker Success Rate:{1 - total_fail/total_episode/num_seekers:.2f}")
    total_time += total_frames/120

print(f"\nTotal Human Control Data Amount: {len(all_data)}")
print(f"Total Heuristic Control Data Amount: {len(all_data1)}")
print(f"Total Collected Time: {total_time/5:.2f} minutes")

#only smaple half from the all_data
ratio = 0.25
random_index = np.random.choice(len(all_data), int(len(all_data) * ratio), replace=False)
all_data = torch.utils.data.Subset(all_data, random_index)

random_index1 = np.random.choice(len(all_data1), int(len(all_data1) * ratio), replace=False)
all_data1 = torch.utils.data.Subset(all_data1, random_index1)

print(f"\nTotal Used Human Control Time: {total_time/5*ratio:.2f} minutes")

batch_size = 128
len_data = len(all_data)
train_data = torch.utils.data.Subset(all_data, range(0, int(len_data * 0.90)))
val_data = torch.utils.data.Subset(all_data, range(int(len_data * 0.90), len_data ))
trainloader = DataLoader(train_data , batch_size=batch_size, shuffle=True,num_workers=35)
valloader = DataLoader(val_data , batch_size=batch_size, shuffle=True,num_workers=35)

print(f"\nHuman Training data size: {len(train_data)}")
print(f"Human Validation data size: {len(val_data)}")

batch_size = 128
len_data1 = len(all_data1)
train_data1 = torch.utils.data.Subset(all_data1, range(0, int(len_data1 * 0.90)))
val_data1 = torch.utils.data.Subset(all_data1, range(int(len_data1 * 0.90), len_data1 ))
trainloader1 = DataLoader(train_data1 , batch_size=batch_size, shuffle=True,num_workers=35)
valloader1 = DataLoader(val_data1 , batch_size=batch_size, shuffle=True,num_workers=35)

print(f"\nHeuristic Training data size: {len(train_data1)}")
print(f"Heuirstic Validation data size: {len(val_data1)}\n")

#load model
model = CustomResNet18(num_of_frames)
criterion = nn.MSELoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

#Load Encoder Weight
original_first_layer_weights = model.resnet18.conv1.weight.data.cuda()

state_dict = torch.load("Weights/IL_Team/model_epoch19.pth")["model_state_dict"]
updated_state_dict = {}
for key in state_dict.keys():
    if 'fc' not in key:
        updated_state_dict[key] = state_dict[key]

model.load_state_dict(updated_state_dict,strict=False)

first_layer_weights = model.resnet18.conv1.weight.data.cuda()
loaded_first_layer_weights = state_dict['resnet18.conv1.weight'].cuda()

if torch.equal(first_layer_weights, loaded_first_layer_weights) and not torch.equal(loaded_first_layer_weights,original_first_layer_weights):
    print("\nEncoder Weights loaded successfully!")
else:
    print("\nError: Weights not loaded correctly.")

for param in model.resnet18.parameters():
    param.requires_grad = False
print("Encoder Freeze")

save_base_directory = "Weights/FT_team_10minutes"

import sys
sys.setrecursionlimit(10000000)

learning_rate = 0.001
dataset_name = base_folder.split("-1")[-1]
optimizer_name = "Adam"
loss_name = "MSE"

t_l,v_l = fine_tune(model,
          50,
          'cuda',
          trainloader,valloader,
          trainloader1,valloader1,
          criterion,
          optimizer,
          save_base_directory,
          0.5,
)

training_result = {"Training_loss":t_l,"Validation_loss":v_l}
with open(save_base_directory+'/training_result.json', 'w') as f:
    json.dump(training_result, f)





