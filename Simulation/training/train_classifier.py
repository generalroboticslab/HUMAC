from Dataloader.Hide_and_Seek_Dataset_Classification import HideandSeekDataset
import os
from torch.utils.data import ConcatDataset
from torch.utils.data import  DataLoader
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from Model.Resnet import CustomResNet18, classification, evaluate_model
import json
from torch import nn
from torchvision.models import resnet18

seed_value = 42
torch.manual_seed(seed_value)
np.random.seed(seed_value)
import random
random.seed(seed_value)

num_of_frames=5
total_human_control1 = 0

for folder in os.scandir("Dataset_human"):
    
    total_fail = 0
    total_episode = 0
    total_human_control = 0 

    base_folder = os.path.join("Dataset_human", folder.name)
    num_seekers = int(str(folder.name)[0])

    for seed_num in os.scandir(base_folder):
        for agent_id in range(num_seekers):
            dataset = HideandSeekDataset(os.path.join(base_folder,seed_num.name),agent_id)
            
            total_fail += int(dataset.is_seeker_fail())
            total_human_control +=dataset.num_of_human_control()
            
            if (dataset.have_missing_agent()):
                print("BAD")
            
            try:
                all_data = ConcatDataset([all_data,dataset])
            except:
                all_data = dataset
        total_episode += 1

        # if total_episode == 1:
        #     break

    print(f"{folder.name}, Total Episode: {total_episode}, Total human control: {total_human_control} ,Seeker Success Rate:{1 - total_fail/total_episode/num_seekers:.2f}")
    
    total_human_control1 += total_human_control

print(f"\nTotal Data Amount: {len(all_data)}")
print(f"Total Human Control: {total_human_control1}\n")

l1 = len(all_data) - total_human_control1
l2 = total_human_control1

weight = torch.zeros((2)).to('cuda')
for i in range(2):
    weight[i] = (l1+l2)/(2*eval(f"l{i+1}"))

print(weight)


model = CustomResNet18(num_of_frames)
criterion = nn.CrossEntropyLoss(weight = weight )
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

#Load Encoder Weight
original_first_layer_weights = model.resnet18.conv1.weight.data.cuda()

state_dict = torch.load("Weights/Pre-train_Encoder_team_resnet_sorted_action/model_epoch113.pth")["model_state_dict"]
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

#prepare dataloader
batch_size = 128
len_data1 = len(all_data)
train_data = torch.utils.data.Subset(all_data, range(0, int(len_data1 * 0.90)))
val_data = torch.utils.data.Subset(all_data, range(int(len_data1 * 0.90), len_data1 ))
trainloader = DataLoader(train_data , batch_size=batch_size, shuffle=True,num_workers=35)
valloader = DataLoader(val_data , batch_size=batch_size, shuffle=True,num_workers=35)

print(f"\nTraining data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}\n")

save_base_directory = "Weights/Classifier_team"

import sys
sys.setrecursionlimit(10000000)

learning_rate = 0.001
dataset_name = base_folder.split("-1")[-1]
optimizer_name = "Adam"
loss_name = "BCE"

# t_l,v_l = classification(model,
#           50,
#           'cuda',
#           trainloader,valloader,
#           criterion,
#           optimizer,
#           save_base_directory,
#           learning_rate,
#           dataset_name,
#           batch_size,
#           optimizer_name,
#           loss_name,
# )

# training_result = {"Training_loss":t_l,"Validation_loss":v_l}
# with open(save_base_directory+'/training_result.json', 'w') as f:
#     json.dump(training_result, f)

# Assume the model is already loaded and the valloader is prepared
state_dict = torch.load("Weights/Classifier_team/model_epoch4.pth")["model_state_dict"]
model.load_state_dict(state_dict,strict=False)

first_layer_weights = model.resnet18.conv1.weight.data.cuda()
loaded_first_layer_weights = state_dict['resnet18.conv1.weight'].cuda()

if torch.equal(first_layer_weights, loaded_first_layer_weights) and not torch.equal(loaded_first_layer_weights,original_first_layer_weights):
    print("\nEncoder Weights loaded successfully!")
else:
    print("\nError: Weights not loaded correctly.")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
f1,precision,recall = evaluate_model(model, valloader, device)

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')





