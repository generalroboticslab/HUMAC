import os
from torch.utils.data import ConcatDataset
from torch.utils.data import  DataLoader
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils import train, load_data_IL, load_data_team, green, red, reset
from model import IL, PE_N, PE_H
import argparse
import sys
import random 
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.setrecursionlimit(10000)

parser = argparse.ArgumentParser(description="Train the model with specified config")
parser.add_argument('--seed_value', type=int, default=42, help='Seed value for randomness')

parser.add_argument('--batch_size', type=int, default=128, help='Batch size for data loader')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
parser.add_argument('--epochs', type=int, default=150, help='Number of epochs for training')
parser.add_argument('--num_workers', type=int, default=1, help='Number of CPU used for data loader')

parser.add_argument('--num_of_frames', type=int, default=5, help='Number of frames to stack')
parser.add_argument('--step_ahead', type=int, default=5, help='Number of step_ahead for prediction')
parser.add_argument('--data_root_folder', type=str, default="path/to/IL/data", help='Path to IL data')
parser.add_argument('--max_num_seekers', type=int, default=4, help='Max amount of seekers in the game during data collection')

# Parse the arguments
args = parser.parse_args()

#fix seed
torch.manual_seed(args.seed_value)
np.random.seed(args.seed_value)
random.seed(args.seed_value)

#load data
all_data_self = load_data_IL(args.num_of_frames,args.data_root_folder,step_ahead=args.step_ahead)
all_data_teammate = load_data_team(args.num_of_frames,args.max_num_seekers,args.data_root_folder,step_ahead=args.step_ahead,teammate_only=True)

#make Imitation learning data loader
all_data = all_data_self
total_size = len(all_data)
indices = np.random.permutation(total_size)
subset_indices = indices[:int(total_size * 0.20)]
all_data = torch.utils.data.Subset(all_data, subset_indices)
train_data = torch.utils.data.Subset(all_data, range(0, int(len(all_data) * 0.90)))
val_data = torch.utils.data.Subset(all_data, range(int(len(all_data)  * 0.90), len(all_data)))

trainloader_IL = DataLoader(train_data , batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers)
valloader_IL = DataLoader(val_data , batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers)

print(f"\nIL Training data size: {len(train_data)}")
print(f"IL Validation data size: {len(val_data)}\n")

#define learning parameter
learning_rate = args.learning_rate
dataset_name = args.data_root_folder.split("-1")[-1]
optimizer_name = "Adam"
loss_name = "MSE"
criterion = nn.MSELoss()

teammate_prediction = IL(args.num_of_frames,num_seekers = args.max_num_seekers-1).to(device)
teammate_optimizer = optim.Adam(filter(lambda p: p.requires_grad, teammate_prediction.parameters()), lr=learning_rate)

#make teammate dataloader
all_data = all_data_teammate
total_size = len(all_data)
indices = np.random.permutation(total_size)
subset_indices = indices[:int(total_size * 0.20)]
all_data = torch.utils.data.Subset(all_data, subset_indices)
train_data = torch.utils.data.Subset(all_data, range(0, int(len(all_data) * 0.90)))
val_data = torch.utils.data.Subset(all_data, range(int(len(all_data)  * 0.90), len(all_data)))

trainloader_teammate = DataLoader(train_data , batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers)
valloader_teammate = DataLoader(val_data , batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers)


print(f"\nTeammate Training data size: {len(train_data)}")
print(f"Teammate Validation data size: {len(val_data)}\n")

teammate_save_base_directory = f"../model_weights/teammate_prediction_model"

#train teammate prediction model
t_l,v_l = train(teammate_prediction,
        args.epochs,
        'cuda',
        trainloader_teammate,valloader_teammate,
        criterion,
        teammate_optimizer,
        teammate_save_base_directory,
        learning_rate,
        dataset_name,
        args.batch_size,
        optimizer_name,
        loss_name,
)

training_result = {"Training_loss":t_l,"Validation_loss":v_l}
with open(teammate_save_base_directory+'/teammate_prediction_training_result.json', 'w') as f:
    json.dump(training_result, f)

#make the self prediction model
self_prediction = IL(args.num_of_frames).to(device)
self_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self_prediction.parameters()), lr=learning_rate)

self_save_base_directory = f"../model_weights/self_prediction_model"

t_l,v_l = train(self_prediction,
    args.epochs,
    'cuda',
    trainloader_IL,valloader_IL,
    criterion,
    self_optimizer,
    self_save_base_directory,
    learning_rate,
    dataset_name,
    args.batch_size,
    optimizer_name,
    loss_name,
)

training_result = {"Training_loss":t_l,"Validation_loss":v_l}
with open(self_save_base_directory+'/self_prediction_training_result.json', 'w') as f:
    json.dump(training_result, f)

model = PE_H(args.num_of_frames).to(device)

#load self encoder weights
original_first_layer_weights = model.resnet18_1.conv1.weight.data.to(device)
state_dict = torch.load(self_save_base_directory+"/model.pth")["model_state_dict"]
updated_state_dict = {}
for key in state_dict.keys():
    if 'fc' not in key:
        new_key = key.replace('resnet18.', 'resnet18_1.')
        updated_state_dict[new_key] = state_dict[key]

model.load_state_dict(updated_state_dict,strict=False)
first_layer_weights = model.resnet18_1.conv1.weight.data.to(device)
loaded_first_layer_weights = state_dict['resnet18.conv1.weight'].to(device)

if torch.equal(first_layer_weights, loaded_first_layer_weights):
    print(f"{green}Encoder1 Weights loaded successfully!{reset}")
else:
    print(f"{red}Encoder1 Weights not loaded correctly.{reset}")



#load teammate encoder weights
original_first_layer_weights = model.resnet18_2.conv1.weight.data.to(device)
state_dict = torch.load(teammate_save_base_directory+"/model.pth")["model_state_dict"]
updated_state_dict = {}
for key in state_dict.keys():
    if 'fc' not in key:
        new_key = key.replace('resnet18.', 'resnet18_2.')
        updated_state_dict[new_key] = state_dict[key]

model.load_state_dict(updated_state_dict,strict=False)

if torch.equal(first_layer_weights, loaded_first_layer_weights):
    print(f"{green}Encoder2 Weights loaded successfully!{reset}")
else:
    print(f"{red}Encoder2 Weights not loaded correctly.{reset}")


optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
save_base_directory = f"../model_weights/pre_PE_H"

#train IL on the PE_H model
t_l,v_l = train(model,
          args.epochs,
          'cuda',
          trainloader_IL,valloader_IL,
          criterion,
          optimizer,
          save_base_directory,
          learning_rate,
          dataset_name,
          args.batch_size,
          optimizer_name,
          loss_name,
)

training_result = {"Training_loss":t_l,"Validation_loss":v_l}
with open(save_base_directory+'/IL_training_result.json', 'w') as f:
    json.dump(training_result, f)