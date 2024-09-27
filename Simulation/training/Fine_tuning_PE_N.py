from Dataloader.Hide_and_Seek_Dataset_Human import HideandSeekDataset
import os
from torch.utils.data import ConcatDataset
from torch.utils.data import  DataLoader
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from Model.Resnet_PE_N import ResNet18_PE_N,fine_tune
from Model.Resnet import CustomResNet18
import json
from torch import nn
from torchvision.models import resnet18

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

root_folder = "Dataset_human_click_short"

total_time = 0
num_of_frames=5
num_seekers_max = 4

for folder in os.scandir(root_folder):
    
    total_fail = 0
    total_episode = 0
    total_frames = 0


    base_folder = os.path.join(root_folder, folder.name)
    num_seekers = int(str(folder.name)[0])

    for seed_num in os.scandir(base_folder):
        for agent_id in range(num_seekers):
            dataset = HideandSeekDataset(os.path.join(base_folder,seed_num.name),num_seekers,agent_id, human_control=True,collect_decision_freqeuncy=0.2,train_decision_frequency=1)
            
            total_fail += int(dataset.is_seeker_fail())
            
            if (dataset.have_missing_agent()):
                print("BAD")
            
            try:
                all_data = ConcatDataset([all_data,dataset])
            except:
                all_data = dataset
            
            dataset1 = HideandSeekDataset(os.path.join(base_folder,seed_num.name),num_seekers,agent_id, human_control=False,collect_decision_freqeuncy=0.2,train_decision_frequency=1)
            
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
print(f"Total Time: {total_time/5:.2f} minutes")

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

model = ResNet18_PE_N(num_of_frames)
criterion = nn.MSELoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
weight1 = torch.load("Weights/IL_PE-N/model_epoch11.pth")['model_state_dict']
model.load_state_dict(weight1)

teammate_prediction_model = CustomResNet18(num_of_frames,num_seekers=num_seekers_max-1)
weight = torch.load("Weights/teammate_prediction/model_epoch141.pth")['model_state_dict']
teammate_prediction_model.load_state_dict(weight)


batch_size = 128
total_size = len(all_data)
indices = np.random.permutation(total_size)
subset_indices = indices[:int(total_size * 0.20)]

all_data = torch.utils.data.Subset(all_data, subset_indices)

train_data = torch.utils.data.Subset(all_data, range(0, int(len(all_data) * 0.90)))
val_data = torch.utils.data.Subset(all_data, range(int(len(all_data)  * 0.90), len(all_data)))
trainloader = DataLoader(train_data , batch_size=batch_size, shuffle=True,num_workers=35)
valloader = DataLoader(val_data , batch_size=batch_size, shuffle=True,num_workers=35)

print(f"\nTraining data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}\n")


save_base_directory = "Weights/FT_PE-N"

import sys
sys.setrecursionlimit(10000000)

learning_rate = 0.001
dataset_name = base_folder.split("-1")[-1]
optimizer_name = "Adam"
loss_name = "MSE"

t_l,v_l = fine_tune(model,
        teammate_prediction_model,
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





