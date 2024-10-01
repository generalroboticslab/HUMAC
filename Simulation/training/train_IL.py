import os
from torch.utils.data import ConcatDataset
from torch.utils.data import  DataLoader
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils import train, load_data_IL, load_data_team, green, red
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

# Parse the arguments
args = parser.parse_args()

#fix seed
torch.manual_seed(args.seed_value)
np.random.seed(args.seed_value)
random.seed(args.seed_value)

#load data
all_data_self = load_data_IL(args.num_of_frames,args.data_root_folder,step_ahead=args.step_ahead)

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

learning_rate = args.learning_rate
dataset_name = args.data_root_folder.split("-1")[-1]
optimizer_name = "Adam"
loss_name = "MSE"
criterion = nn.MSELoss()



model = IL(args.num_of_frames).to(device)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

save_base_directory = f"../model_weights/IL"

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