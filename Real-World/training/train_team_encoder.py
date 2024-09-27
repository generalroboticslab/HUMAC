from Dataloader.Hide_and_Seek_Dataset_real_team import HideandSeekDataset
import os
from torch.utils.data import ConcatDataset
from torch.utils.data import  DataLoader
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from Model.Imitation_learning_team import CNN, train
from Model.Resnet import CustomResNet18
import json

root_folder = "data"

num_of_frames= 3
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
        # print(base_folder,seed_num.name)
        num_traj += 1
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
  
print(f"Total Training Data Amount: {len(all_training_data)}")
print(f"Total Validation Data Amount: {len(all_validation_data)}")

model = CustomResNet18(num_of_frames,num_seekers=3,rotation=False)
criterion = nn.MSELoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)


batch_size = 512

trainloader = DataLoader(all_training_data , batch_size=batch_size, shuffle=True,num_workers=35)
valloader = DataLoader(all_validation_data , batch_size=batch_size, shuffle=True,num_workers=35)

print(f"\nTraining data size: {len(all_training_data)}")
print(f"Validation data size: {len(all_validation_data)}\n")

save_base_directory = f"Weights/Team_Encoder_{num_of_frames}frames_{step_ahead}step_ahead"

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





