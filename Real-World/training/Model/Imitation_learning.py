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
    def __init__(self,num_of_frames):
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
        self.fc3 = nn.Linear(64, 2)


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
        
        for images, targets,_,_ in train_loader:
            # print(images.shape)
            # print(targets.shape)
            images, targets = images.to(device), targets.to(device)
            # Forward pass
            outputs = model(images)
            # Compute the loss
            # print("out"outputs)
            # print(targets)
            # break
            loss = criterion(outputs, targets.squeeze())
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            #make a progress bar for training
            


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
            for images1, targets1,_,_ in val_loader:
                images1, targets1 = images1.to(device), targets1.to(device)
                # Forward pass
                outputs1 = model(images1)
                # Compute the validation loss
                loss1 = criterion(outputs1, targets1.squeeze())
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


def train_on_one(model,
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
        
        for images, targets,_,_ in train_loader:
            indices = torch.nonzero(targets[:, -1] == 1)[:,0].squeeze()
            # print(indices)
    
            # Select only the images and targets where the mask is True
            images = images[indices,:,:,:]
            targets = targets[indices, :2]  # Only take the first two elements (x, y) for the remaining targets
            
            # Skip the batch if no valid targets are found
            if images.size(0) == 0:
                continue
            
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute the loss
            loss = criterion(outputs, targets.squeeze())
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            #make a progress bar for training
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
            for images1, targets1,_,_ in val_loader:
                indices1 = torch.nonzero(targets1[:, -1] == 1)[:,0].squeeze()
        
                # Select only the images and targets where the mask is True
                images1 = images1[indices1,:,:,:]
                targets1 = targets1[indices1, :2]  # Only take the first two elements (x, y) for the remaining targets
                
                # Skip the batch if no valid targets are found
                if images1.size(0) == 0:
                    continue
                
                images1, targets1 = images1.to(device), targets1.to(device)
                
                # Forward pass
                outputs1 = model(images1)
                
                # Compute the validation loss
                loss1 = criterion(outputs1, targets1.squeeze())
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


def train_with_rotation(model,
          num_epochs,
          device,
          train_loader,val_loader,
          criterion1,
          criterion2,
          optimizer,
          checkpoint_dir,
          learning_rate,
          dataset_name,
          batch_size,
          optimizer_name,
          loss_name,
          ratio):
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
        total_mse_loss = 0.0
        total_ce_loss = 0.0
        counter = 1
        
        for images, targets,_,_ in train_loader:
            # print(images.shape)
            images, targets = images.to(device), targets.to(device)
            # Forward pass
            outputs = model(images)
            # Compute the loss
            mse_loss = criterion1(outputs[:,:2], targets[:, :2].squeeze())
            ce_loss = criterion2(outputs[:,2:], targets[:, -1].long().squeeze())
            loss = mse_loss + ratio * ce_loss
            # Backpropagation and optimization
            optimizer.zero_grad()  # Zero the gradients for the first optimizer
            loss.backward()  # Retain the graph to allow another backward pass
            optimizer.step()
            
            
            # Accumulate the total loss
            total_loss += (mse_loss.item() + ce_loss.item())
            total_mse_loss += mse_loss.item()
            total_ce_loss += ce_loss.item()
            #make a progress bar for training
        
            if counter % 2000 == 0:
            
                print(f"Epoch [{epoch+1}/{num_epochs}], Iteration [{counter}/{len(train_loader)}] - Loss {total_loss/counter:.4f}")
            counter += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        average_loss = total_loss / len(train_loader)
        train_loss.append(average_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {average_loss:.4f} - MSE Loss: {total_mse_loss/len(train_loader):.4f} - CE Loss: {total_ce_loss/len(train_loader):.4f}")
        
        torch.cuda.empty_cache()

        model.eval()
        val_loss = 0.0
        val_mse_loss = 0.0
        val_ce_loss = 0.0
        with torch.no_grad():
            for images1, targets1,_,_ in val_loader:
                images1, targets1 = images1.to(device), targets1.to(device)
                # Forward pass
                outputs1 = model(images1)
                # Compute the validation loss
                mse_loss = criterion1(outputs1[:,:2], targets1[:, :2].squeeze())
                ce_loss = criterion2(outputs1[:,2:], targets1[:, -1].long().squeeze())
                
                val_mse_loss += mse_loss.item()
                val_ce_loss += ce_loss.item()
                loss1 = mse_loss + ratio*ce_loss
                #loss1 = criterion1(outputs1[:,:2], targets1[:, :2].squeeze()) + criterion2(outputs1[:,2:], targets1[:, -1].long().squeeze())
                val_loss += loss1.item()
        average_val_loss = val_loss / len(val_loader)
        val_loss_list.append(average_val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {average_val_loss:.4f} - MSE Loss: {val_mse_loss/len(val_loader):.4f} - CE Loss: {val_ce_loss/len(val_loader):.4f}")
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
