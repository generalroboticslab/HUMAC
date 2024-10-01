import os
from torch.utils.data import ConcatDataset
from torch.utils.data import  DataLoader
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils import fine_tune, load_data_human, green, red, reset
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
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
parser.add_argument('--num_workers', type=int, default=1, help='Number of CPU used for data loader')

parser.add_argument('--num_of_frames', type=int, default=5, help='Number of frames to stack')
parser.add_argument('--step_ahead', type=int, default=5, help='Number of step_ahead for prediction')
parser.add_argument('--data_root_folder', type=str, default="path/to/IL/data", help='Path to IL data')
parser.add_argument('--max_num_seekers', type=int, default=4, help='Max amount of seekers in the game during data collection')
parser.add_argument('--model', type=str, default="IL", help='Model to fine-tune')

# Parse the arguments
args = parser.parse_args()

#fix seed
torch.manual_seed(args.seed_value)
np.random.seed(args.seed_value)
random.seed(args.seed_value)

#load data
all_data_human, all_data_heuristic = load_data_human(args.num_of_frames,args.data_root_folder,step_ahead=args.step_ahead)

all_data = all_data_human
train_data = torch.utils.data.Subset(all_data, range(0, int(len(all_data) * 0.90)))
val_data = torch.utils.data.Subset(all_data, range(int(len(all_data)  * 0.90), len(all_data)))

trainloader_human = DataLoader(train_data , batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers)
valloader_human = DataLoader(val_data , batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers)

print(f"\nHuman Training data size: {len(train_data)}")
print(f"Human Validation data size: {len(val_data)}\n")

all_data_human, all_data_heuristic = load_data_human(args.num_of_frames,args.data_root_folder,step_ahead=args.step_ahead)

all_data = all_data_heuristic
train_data = torch.utils.data.Subset(all_data, range(0, int(len(all_data) * 0.90)))
val_data = torch.utils.data.Subset(all_data, range(int(len(all_data)  * 0.90), len(all_data)))

trainloader_heuristic = DataLoader(train_data , batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers)
valloader_heuristic = DataLoader(val_data , batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers)

print(f"\nHeuirstic Training data size: {len(train_data)}")
print(f"Heuristic Validation data size: {len(val_data)}\n")

learning_rate = args.learning_rate
dataset_name = args.data_root_folder.split("-1")[-1]
optimizer_name = "Adam"
loss_name = "MSE"
criterion = nn.MSELoss()

if args.model != "PE_N":
    if args.model == "PE_T":
        model = IL(args.num_of_frames).to(device)
    else:
        model = eval(args.model)(args.num_of_frames).to(device)
else:
    model = eval(args.model)(args.num_of_frames,max_num_teammate = args.max_num_seekers-1).to(device)

if args.model == "IL":
    weights = torch.load(f"../model_weights/{args.model}/model.pth")["model_state_dict"]
else:
    weights = torch.load(f"../model_weights/pre_{args.model}/model.pth")["model_state_dict"]

model.load_state_dict(weights,strict=True)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)


if args.model == "IL":
    save_base_directory = f"../model_weights/{args.model}-FT"
else:
    save_base_directory = f"../model_weights/{args.model}"


t_l,v_l = fine_tune(model,
          args.epochs,
          'cuda',
          trainloader_human,valloader_human,
          trainloader_heuristic,valloader_heuristic,
          criterion,
          optimizer,
          save_base_directory,
          ratio=0.5
)

training_result = {"Training_loss":t_l,"Validation_loss":v_l}
with open(save_base_directory+'/fine-tune_result.json', 'w') as f:
    json.dump(training_result, f)