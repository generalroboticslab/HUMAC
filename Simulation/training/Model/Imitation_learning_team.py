import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time
import os
import wandb

class CNN(nn.Module):
    def __init__(self,num_of_frames,num_seeker_max):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=num_of_frames*4, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
    
        # self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        # self.bn4 = nn.BatchNorm2d(128)
        # self.relu4 = nn.ReLU()
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
    
        self.fc1 = nn.Linear(512, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(64, 2*num_seeker_max)


    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)

        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        output = self.fc3(x) 

        return output

def train(model,
          num_epochs,
          device,
          train_loader,val_loader,
          criterion,
          optimizer,
          checkpoint_dir,
          learning_rate,
          dataset_name,
          batch_size,
          optimizer_name,
          loss_name,):
    wandb.init(
        # set the wandb project where this run will be logged
        project="Multi-agent-HS",
        name = checkpoint_dir.split("/")[-1], 
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": learning_rate,
        "architecture": type(model).__name__,
        "Dataset": dataset_name,
        "epochs": num_epochs,
        "Batch_size": batch_size,
        "optimizer": optimizer_name,
        "Loss":loss_name,
        }
    )
    torch.cuda.empty_cache()
    train_loss = []
    val_loss_list = []
    model = model.to(device)
    best_val_loss = float('inf')

    if not os.path.exists(checkpoint_dir):
        # If it doesn't exist, create it
        os.makedirs(checkpoint_dir)

    
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        counter = 1
        
        for images, targets,_ in train_loader:
            images, targets = images.to(device), targets.to(device)
            # Forward pass
            outputs = model(images).squeeze()
            # Compute the loss
            targets = targets.squeeze()

            mask = (targets != 50).squeeze()

            # Apply the mask to outputs and targets
            filtered_outputs = outputs[mask]
            filtered_targets = targets[mask]


            loss = criterion(filtered_outputs , filtered_targets)
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            

            if counter % 2000 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Iteration [{counter}/{len(train_loader)}] - Loss {total_loss/counter:.4f}")
            
            
            counter += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        average_loss = total_loss / len(train_loader)
        train_loss.append(average_loss)


        print(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {average_loss:.4f}")
        
        torch.cuda.empty_cache()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images1, targets1,_ in val_loader:
                images1, targets1 = images1.to(device), targets1.to(device)
                # Forward pass
                targets1 = targets1.squeeze()
                outputs1 = model(images1).squeeze()

                mask1 = (targets1 != 50).squeeze()

                filtered_outputs1 = outputs1[mask1]
                filtered_targets1 = targets1[mask1]            
                # Compute the validation loss
                loss1 = criterion(filtered_outputs1, filtered_targets1)
                val_loss += loss1.item()
        average_val_loss = val_loss / len(val_loader)
        val_loss_list.append(average_val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {average_val_loss:.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}] - Time: {elapsed_time:.4f}s")
        
        torch.cuda.empty_cache()
        wandb.log({"Training loss": average_loss,"Validation loss": average_val_loss})

        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_path)
            print(f'Model saved at epoch {epoch+1}')
        print('\n')
    return train_loss,val_loss_list
