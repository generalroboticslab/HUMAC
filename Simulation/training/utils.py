from torch import nn
from torchvision.models import resnet18
import torch.nn.functional as F
import torch 
import os
import time
import wandb
from itertools import zip_longest
from torch.utils.data import ConcatDataset
from torch.utils.data import  DataLoader
import Dataloader.Hide_and_Seek_Dataset_IL as HideandSeekDataset_IL
import Dataloader.Hide_and_Seek_Dataset_IL_team as HideandSeekDataset_IL_team
import Dataloader.Hide_and_Seek_Dataset_Human as HideandSeekDataset_Human

green = '\033[92m'
red = '\033[91m'
blue = '\033[94m'
yellow = '\033[93m'
reset = '\033[0m'

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
            checkpoint_path = os.path.join(checkpoint_dir, f'model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_path)
            print(f'Model saved at epoch {epoch+1}')
        print('\n')
    return train_loss,val_loss_list


def fine_tune(model,num_epochs,device,train_loader,val_loader,old_train_loader,old_val_loader,criterion,optimizer,checkpoint_dir,ratio):
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="Multi-agent-HS",
        name = checkpoint_dir.split("/")[-1], 
        
        # track hyperparameters and run metadata
    )
    run.tags = ["Fine Tuning"]

    torch.cuda.empty_cache()
    train_loss = []
    val_loss_list = []
    model = model.to(device)
    best_val_loss = float('inf')

    if not os.path.exists(checkpoint_dir):
        # If it doesn't exist, create it
        os.makedirs(checkpoint_dir)


    for epoch in range(num_epochs):

        big_train_loader = zip_longest(train_loader,old_train_loader)
        start_time = time.time()
        model.train()
        total_loss = 0.0
        counter = 1

        total_old_loss = 0
        total_new_loss = 0
        
        for train_data,old_train_data in big_train_loader:
            if counter > len(train_loader):
                break

            images1, targets1,_ = train_data
            images, targets = images1.to(device), targets1.to(device)
            outputs = model(images)
            new_loss = criterion(outputs, targets.squeeze())


            old_images1, old_targets1,_ = old_train_data
            old_images, old_targets = old_images1.to(device), old_targets1.to(device)
            old_outputs = model(old_images)
            old_loss = criterion(old_outputs, old_targets.squeeze())
                            
            
            loss = ratio*new_loss + (1-ratio)*old_loss
            
            total_old_loss += old_loss
            total_new_loss += new_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            if counter % 800 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Iteration [{counter}/{len(train_loader)}] - Loss {total_loss/counter:.4f}")
                print(f"Old loss: {total_old_loss/counter:.4f}, New loss: {total_new_loss/counter:.4f}")
            counter += 1
    

        end_time = time.time()
        elapsed_time = end_time - start_time
        average_loss = total_loss / (2*len(train_loader)) ## two dataset 
        train_loss.append(average_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {average_loss:.4f}")
        
        torch.cuda.empty_cache()


        model.eval()
        
        big_val_loader = zip_longest(val_loader,old_val_loader)
        val_loss = 0.0
        total_old_loss1 = 0
        total_new_loss1 = 0 

        with torch.no_grad():
            for val_data,old_val_data in big_val_loader:
                if val_data is None:
                    break
                images2, targets2,_ =  val_data
                images3, targets3,_ =  old_val_data

                images1, targets1 = images2.to(device), targets2.to(device)
                outputs1 = model(images1)
                new_loss1 = criterion(outputs1, targets1.squeeze())

                old_images1, old_targets1 = images3.to(device), targets3.to(device)
                old_outputs1 = model(old_images1)
                old_loss1 = criterion(old_outputs1, old_targets1.squeeze())

                val_loss += ratio*new_loss1.item() + (1-ratio)*old_loss1.item()
                total_old_loss1 += old_loss1.item()
                total_new_loss1 += new_loss1.item()

        average_val_loss = val_loss / len(val_loader)
        val_loss_list.append(average_val_loss)

        print(f"Old Validation loss: {total_old_loss1/len(val_loader):.4f}, New Validation loss: {total_new_loss1/len(val_loader):.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {average_val_loss:.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}] - Time: {elapsed_time:.4f}s")

        wandb.log({"Training loss": average_loss,
                "Validation loss": average_val_loss,
                "Old Training loss":total_old_loss/counter,
                "New Training loss":total_new_loss/counter,
                "Old Validation loss":total_old_loss1/len(val_loader),
                "New Validation loss":total_new_loss1/len(val_loader),
                })

        torch.cuda.empty_cache()

        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, checkpoint_path)
            print(f'Model saved at epoch {epoch+1}')
        print('\n')
    
    wandb.finish()

    return train_loss,val_loss_list

def load_data_IL(num_of_frames,root_folder,step_ahead=5):
    for folder in os.scandir(root_folder):  
        total_fail = 0
        total_episode = 0

        base_folder = os.path.join(root_folder, folder.name)
        num_seekers = int(str(folder.name)[0])

        for seed_num in os.scandir(base_folder):

            for agent_id in range(num_seekers):
                dataset = HideandSeekDataset_IL.HideandSeekDataset(os.path.join(base_folder,seed_num.name),agent_id,num_seekers,step_ahead=step_ahead)
                
                total_fail += int(dataset.is_seeker_fail())
                
                if (dataset.have_missing_agent()):
                    print(f"{folder.name},{seed_num.name},{agent_id} has missing data")
                
                try:
                    all_data = ConcatDataset([all_data,dataset])
                except:
                    all_data = dataset

            total_episode += 1

        # print(f"{folder.name}, Total Episode: {total_episode}, Seeker Success Rate:{1 - total_fail/total_episode/num_seekers:.2f}")
    return all_data


def load_data_team(num_of_frames,num_seekers_max,root_folder,step_ahead=5,teammate_only=True):
    for folder in os.scandir(root_folder):
        
        total_fail = 0
        total_episode = 0

        base_folder = os.path.join(root_folder, folder.name)
        num_seekers = int(str(folder.name)[0])


        for seed_num in os.scandir(base_folder):
            for agent_id in range(num_seekers):
                dataset = HideandSeekDataset_IL_team.HideandSeekDataset(os.path.join(base_folder,seed_num.name),agent_id,num_seekers,num_seekers_max,step_ahead=step_ahead,teammate_only=teammate_only)
                total_fail += int(dataset.is_seeker_fail())
                
                if (dataset.have_missing_agent()):
                    print(f"{folder.name},{seed_num.name},{agent_id} has missing data")
                
                try:
                    all_data = ConcatDataset([all_data,dataset])
                except:
                    all_data = dataset
            total_episode += 1

        # print(f"{folder.name}, Total Episode: {total_episode}, Seeker Success Rate:{1 - total_fail/total_episode/num_seekers:.2f}")
    return all_data


def load_data_human(num_of_frames,root_folder,step_ahead=5):
    for folder in os.scandir(root_folder):
    
        total_fail = 0
        total_episode = 0
        total_frames = 0


        base_folder = os.path.join(root_folder, folder.name)
        num_seekers = int(str(folder.name)[0])


        for seed_num in os.scandir(base_folder):
            for agent_id in range(num_seekers):
                dataset = HideandSeekDataset.HideandSeekDataset(os.path.join(base_folder,seed_num.name),num_seekers,agent_id, human_control=True,step_ahead=step_ahead)
                
                total_fail += int(dataset.is_seeker_fail())
                
                if (dataset.have_missing_agent()):
                    print(f"{folder.name},{seed_num.name},{agent_id} has missing data")
                try:
                    all_data = ConcatDataset([all_data,dataset])
                except:
                    all_data = dataset
                
                dataset1 = HideandSeekDataset.HideandSeekDataset(os.path.join(base_folder,seed_num.name),num_seekers,agent_id, human_control=False,step_ahead=step_ahead)
                
                total_fail += int(dataset1.is_seeker_fail())
                
                if (dataset1.have_missing_agent()):
                    print(f"{folder.name},{seed_num.name},{agent_id} has missing data")

                
                try:
                    all_data1 = ConcatDataset([all_data1,dataset1])
                except:
                    all_data1 = dataset1


            total_episode += 1

            total_frames += dataset.total_len()

        # print(f"{folder.name}, Total Episode: {total_episode}, Seeker Success Rate:{1 - total_fail/total_episode/num_seekers:.2f}")
    
    return all_data,all_data1


