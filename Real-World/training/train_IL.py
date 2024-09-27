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
import random

root_folder = "data"

num_of_frames = 3
step_ahead = 5

random.seed(0)

# Randomly sample 20 integers from 1 to 50 with fixed seed
sampled_integers_fixed_seed = random.sample(range(1, 51), 50)

for folder in os.scandir(root_folder):
    
    total_fail = 0
    total_episode = 0
    total_time = 0
    counter = 0
    base_folder = os.path.join(root_folder, folder.name)
    num_seekers = int(str(folder.name)[0])

    # if num_seekers != 3:
    #     continue
    for seed_num in os.scandir(base_folder):
        counter += 1
        if counter <= 30 or counter > 50:
            continue
        # print(base_folder,seed_num.name)
        for agent_id in range(0,num_seekers):
            dataset = HideandSeekDataset(os.path.join(base_folder,seed_num.name),agent_id,num_seekers,num_of_frame=num_of_frames,step_ahead=step_ahead)
            if (dataset.have_missing_agent()):
                print("BAD")
            
            try:
                all_data = ConcatDataset([all_data,dataset])
            except:
                all_data = dataset
            
        total_episode += 1
        total_fail += int(dataset.is_seeker_fail())
        total_time += int(len(dataset))
        # dataset.plot(10)
        # break
 

    print(f"{folder.name}, Total Episode: {total_episode}, Seeker Success Rate:{total_episode-total_fail,total_episode}%, Average Time: {total_time/5/total_episode:.2f}s")
print(f"Total Data Amount: {len(all_data)}")


# model = resnet18(weights=False)
# model.conv1 = nn.Conv2d(4*num_of_frames,64,kernel_size=7,stride =2, padding=3, bias=False)
# model.fc = nn.Linear(512,2)
# model = CustomResNet18(num_of_frames)


# criterion = nn.MSELoss()
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)


# batch_size = 128

# train_data = torch.utils.data.Subset(all_data, range(0, int(len(all_data) * 0.90)))
# val_data = torch.utils.data.Subset(all_data, range(int(len(all_data)  * 0.90), len(all_data)  ))

# trainloader = DataLoader(train_data , batch_size=batch_size, shuffle=True,num_workers=35)
# valloader = DataLoader(val_data , batch_size=batch_size, shuffle=True,num_workers=35)

# print(f"\nTraining data size: {len(train_data)}")
# print(f"Validation data size: {len(val_data)}\n")


# save_base_directory = "Weights/IL_Long_real"

# import sys
# sys.setrecursionlimit(10000000)

# learning_rate = 0.001
# dataset_name = base_folder.split("-1")[-1]
# optimizer_name = "Adam"
# loss_name = "MSE"

# t_l,v_l = train(model,
#           150,
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
