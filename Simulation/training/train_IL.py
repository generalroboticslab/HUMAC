from Dataloader.Hide_and_Seek_Dataset_IL import HideandSeekDataset
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

seed_value = 42
torch.manual_seed(seed_value)
np.random.seed(seed_value)
import random
random.seed(seed_value)

root_folder = "Dataset_short"

info_dirtionaty = {}
list1 = ["1Seeker_vs_1Hider","2Seeker_vs_1Hider","2Seeker_vs_2Hider","3Seeker_vs_1Hider","3Seeker_vs_2Hider","3Seeker_vs_3Hider","4Seeker_vs_1Hider","4Seeker_vs_2Hider","4Seeker_vs_3Hider","4Seeker_vs_4Hider"]
range_list = [list(range(1001,1801)),list(range(1801,2601)),list(range(2601,3401)),list(range(3401,4201)),list(range(4201,5001)),list(range(5001,5801)),list(range(5801,6601)),list(range(6601,7401)),list(range(7401,8201)),list(range(8201,9001))]

for i,j in zip(list1,range_list):
    info_dirtionaty[i] = j

num_of_frames=5
for folder in os.scandir(root_folder):
    # print(folder.name)     
    total_fail = 0
    total_episode = 0

    base_folder = os.path.join(root_folder, folder.name)
    num_seekers = int(str(folder.name)[0])

    for seed_num in os.scandir(base_folder):
        
        info_dirtionaty[folder.name].remove(int(seed_num.name.split("=")[-1]))
        
        for agent_id in range(num_seekers):
            dataset = HideandSeekDataset(os.path.join(base_folder,seed_num.name),agent_id,num_seekers)
            
            total_fail += int(dataset.is_seeker_fail())
            
            if (dataset.have_missing_agent()):
                print("BAD")
            
            try:
                all_data = ConcatDataset([all_data,dataset])
            except:
                all_data = dataset
            # dataset.plot(20)
            # break
        total_episode += 1

        # if total_episode == 1:
        #     break
    print(f"{folder.name}, Total Episode: {total_episode}, Seeker Success Rate:{1 - total_fail/total_episode/num_seekers:.2f}")
    print(f"Missing Seed: {info_dirtionaty[folder.name]}")
    # break
print(f"Total Data Amount: {len(all_data)}")

model = resnet18(pretrained=False)
model.conv1 = nn.Conv2d(4*num_of_frames,64,kernel_size=7,stride =2, padding=3, bias=False)
model.fc = nn.Linear(512,2)
model = CustomResNet18(num_of_frames)


criterion = nn.MSELoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)


batch_size = 128

total_size = len(all_data)
indices = np.random.permutation(total_size)
subset_indices = indices[:int(total_size * 0.20)]

all_data = torch.utils.data.Subset(all_data, subset_indices)

train_data = torch.utils.data.Subset(all_data, range(0, int(len(all_data) * 0.90)))
val_data = torch.utils.data.Subset(all_data, range(int(len(all_data)  * 0.90), len(all_data)  ))

trainloader = DataLoader(train_data , batch_size=batch_size, shuffle=True,num_workers=100)
valloader = DataLoader(val_data , batch_size=batch_size, shuffle=True,num_workers=100)

print(f"\nTraining data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}\n")

save_base_directory = "Weights/Imitation_Learning_resnet_short"

import sys
sys.setrecursionlimit(10000000)

learning_rate = 0.001
dataset_name = base_folder.split("-1")[-1]
optimizer_name = "Adam"
loss_name = "MSE"

t_l,v_l = train(model,
          100,
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





