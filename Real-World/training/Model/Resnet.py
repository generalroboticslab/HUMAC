from torch import nn
from torchvision.models import resnet18
import torch.nn.functional as F
import torch 
import os
import time
import wandb
from itertools import zip_longest

green = '\033[92m'
red = '\033[91m'
blue = '\033[94m'
yellow = '\033[93m'
reset = '\033[0m'

class CustomResNet18(nn.Module):
    def __init__(self, num_of_frames,num_seekers=1,rotation = False):
        super(CustomResNet18, self).__init__()
        # Load a pre-trained ResNet18 model
        self.rotation = rotation
        self.resnet18 = resnet18(pretrained=False)
        
        # Replace the first convolution layer
        self.resnet18.conv1 = nn.Conv2d(4 * num_of_frames, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the original fully connected layer
        self.resnet18.fc = nn.Identity()
        
        # Add three new fully connected layers
        self.fc1 = nn.Linear(512, 256)  # First FC layer
        self.fc2 = nn.Linear(256, 128)        # Second FC layer
        self.fc3 = nn.Linear(128, 2*num_seekers)  # Third FC layer

        self.fc4 = nn.Linear(512, 256)  # First FC layer
        self.fc5 = nn.Linear(256, 128)        # Second FC layer
        self.fc6 = nn.Linear(128, 3) 

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Forward pass through the modified ResNet18 up to the original FC layer
        latent = self.resnet18(x)
        
        # Forward pass through the new FC layers
        x = F.relu(self.fc1(latent))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) 
        if self.rotation:
            y = F.relu(self.fc4(latent))
            y = F.relu(self.fc5(y))
            y = self.softmax(self.fc6(y))
            
        return torch.cat((x,y),dim=1) if self.rotation else x
    
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
    # best_epoch = -1
    # best_old_val_loss = float('inf')
    # best_new_val_loss = float('inf')


    if not os.path.exists(checkpoint_dir):
        # If it doesn't exist, create it
        os.makedirs(checkpoint_dir)

    print("\nTraining Process:")


    for epoch in range(num_epochs):

        # print(f"Best Epoch: {blue}{best_epoch}{reset}, Best Validation Loss: {yellow}{best_val_loss:.2f}{reset}, Progress: [{green}{epoch*'>'}{reset}{(num_epochs- epoch)*'-'}]", end="\r")

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

            images1, targets1,_,_ = train_data
            images, targets = images1.to(device), targets1.to(device)
            outputs = model(images)
            new_loss = criterion(outputs, targets.squeeze())


            old_images1, old_targets1,_,_ = old_train_data
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
                images2, targets2,_,_ =  val_data
                images3, targets3,_,_ =  old_val_data

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
            best_epoch = epoch 
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, checkpoint_path)
            print(f'Model saved at epoch {epoch+1}')
        print("\n")
    wandb.finish()

    return train_loss,val_loss_list

