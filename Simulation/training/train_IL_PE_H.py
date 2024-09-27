from Dataloader.Hide_and_Seek_Dataset_IL import HideandSeekDataset
import os
from torch.utils.data import ConcatDataset
from torch.utils.data import  DataLoader
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from Model.Resnet_PE_H import ResNet18_PE_H,train
import json

seed_value = 42
torch.manual_seed(seed_value)
np.random.seed(seed_value)
import random
random.seed(seed_value)

root_folder = "Dataset_short"

num_of_frames=5
for folder in os.scandir(root_folder):
    
    total_fail = 0
    total_episode = 0

    base_folder = os.path.join(root_folder, folder.name)
    num_seekers = int(str(folder.name)[0])

    # if num_seekers != 1:
    for seed_num in os.scandir(base_folder):
        for agent_id in range(num_seekers):
            dataset = HideandSeekDataset(os.path.join(base_folder,seed_num.name),agent_id,num_seekers)
            
            total_fail += int(dataset.is_seeker_fail())
            
            if (dataset.have_missing_agent()):
                print("BAD")
            
            try:
                all_data = ConcatDataset([all_data,dataset])
            except:
                all_data = dataset
            
            # dataset.plot(2)
            # break
        total_episode += 1

        # if total_episode == 1:
        #     break

    print(f"{folder.name}, Total Episode: {total_episode}, Seeker Success Rate:{1 - total_fail/total_episode/num_seekers:.2f}")
# break    

print(f"Total Data Amount: {len(all_data)}")

model = ResNet18_PE_H(num_of_frames)
criterion = nn.MSELoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)


#load Encoder 1
original_first_layer_weights = model.resnet18_1.conv1.weight.data.cuda()

state_dict = torch.load("Weights/Imitation_Learning_resnet_short/model_epoch100.pth")["model_state_dict"]
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

state_dict = torch.load("Weights/teammate_prediction_short/model_epoch44.pth")["model_state_dict"]
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

save_base_directory = "Weights/Imitation_learning_PE-H"

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





