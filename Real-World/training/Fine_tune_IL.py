from Dataloader.Hide_and_Seek_Dataset_real_Human_direction import HideandSeekDataset
import os
from torch.utils.data import ConcatDataset
from torch.utils.data import  DataLoader
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from Model.Resnet import CustomResNet18, fine_tune
import json
from torch import nn

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

root_folder = "data_human"

num_of_frames = 3
step_ahead = 15
for folder in os.scandir(root_folder):
    
    total_fail = 0
    total_episode = 0
    total_time = 0

    base_folder = os.path.join(root_folder, folder.name)
    num_seekers = int(str(folder.name)[0])

    # if num_seekers != 3:
    #     continue
    for seed_num in os.scandir(base_folder):
        # print(base_folder,seed_num.name)
        for agent_id in range(0,num_seekers):
            dataset = HideandSeekDataset(os.path.join(base_folder,seed_num.name),agent_id,num_seekers,num_of_frame=num_of_frames,human_control = True,step_ahead=step_ahead)
            if (dataset.have_missing_agent()):
                print("BAD")
            
            try:
                all_data = ConcatDataset([all_data,dataset])
            except:
                all_data = dataset
        
            dataset1 = HideandSeekDataset(os.path.join(base_folder,seed_num.name), agent_id,num_seekers,num_of_frame=num_of_frames, human_control=False,step_ahead=step_ahead)
            

            if (dataset1.have_missing_agent()):
                print("BAD")
            
            try:
                all_data1 = ConcatDataset([all_data1,dataset1])
            except:
                all_data1 = dataset1

        total_episode += 1
        total_fail += int(dataset.is_seeker_fail())
        # total_time += int(len(dataset)/num_seekers)
    #     dataset.plot(-1)
    #     dataset1.plot(30)
    #     break
        
 
    # break
    print(f"{folder.name}, Total Episode: {total_episode}, Seeker Success Rate:{100 - total_fail/total_episode*100:.2f}%")


print(f"\nTotal Human Control Data Amount: {len(all_data)}")
print(f"Total Heuristic Control Data Amount: {len(all_data1)}")
print(f"Total Collected Time: {total_time/10:.2f} minutes")


# batch_size = 512
# len_data = len(all_data)
# train_data = torch.utils.data.Subset(all_data, range(0, int(len_data * 0.90)))
# val_data = torch.utils.data.Subset(all_data, range(int(len_data * 0.90), len_data ))
# trainloader = DataLoader(train_data , batch_size=batch_size, shuffle=True,num_workers=35)
# valloader = DataLoader(val_data , batch_size=batch_size, shuffle=True,num_workers=35)

# print(f"\nHuman Training data size: {len(train_data)}")
# print(f"Human Validation data size: {len(val_data)}")

# len_data1 = len(all_data1)
# train_data1 = torch.utils.data.Subset(all_data1, range(0, int(len_data1 * 0.90)))
# val_data1 = torch.utils.data.Subset(all_data1, range(int(len_data1 * 0.90), len_data1 ))
# trainloader1 = DataLoader(train_data1 , batch_size=batch_size, shuffle=True,num_workers=35)
# valloader1 = DataLoader(val_data1 , batch_size=batch_size, shuffle=True,num_workers=35)

# print(f"\nHeuristic Training data size: {len(train_data1)}")
# print(f"Heuirstic Validation data size: {len(val_data1)}\n")

# #load model
# model = CustomResNet18(num_of_frames,rotation=False)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# #Load Encoder Weight
# original_first_layer_weights = model.resnet18.conv1.weight.data.cuda()

# state_dict = torch.load("Weights/Team_IL_3frames_15step_ahead_direction/model_epoch2.pth")["model_state_dict"]
# updated_state_dict = {}
# for key in state_dict.keys():
#     if 'fc' not in key:
#         updated_state_dict[key] = state_dict[key]

# # model.load_state_dict(updated_state_dict,strict=False)
# model.load_state_dict(state_dict,strict=False)

# first_layer_weights = model.resnet18.conv1.weight.data.cuda()
# loaded_first_layer_weights = state_dict['resnet18.conv1.weight'].cuda()

# if torch.equal(first_layer_weights, loaded_first_layer_weights) and not torch.equal(loaded_first_layer_weights,original_first_layer_weights):
#     print(f"\n{green}Encoder Weights loaded successfully{reset}")
# else:
#     print(f"\n{red}Error: Weights not loaded correctly{reset}")

# for param in model.resnet18.parameters():
#     param.requires_grad = False
# print(f"{blue}Encoder Freezed{reset}")

# save_base_directory = f"Weights/FT_Team_{num_of_frames}frames_{step_ahead}step_ahead_direction"

# import sys
# sys.setrecursionlimit(10000000)

# learning_rate = 0.001
# dataset_name = base_folder.split("-1")[-1]
# optimizer_name = "Adam"
# loss_name = "MSE"

# t_l,v_l = fine_tune(model,
#           50,
#           'cuda',
#           trainloader,valloader,
#           trainloader1,valloader1,
#           criterion,
#           optimizer,
#           save_base_directory,
#           0.5,
# )

# training_result = {"Training_loss":t_l,"Validation_loss":v_l}
# with open(save_base_directory+'/training_result.json', 'w') as f:
#     json.dump(training_result, f)





