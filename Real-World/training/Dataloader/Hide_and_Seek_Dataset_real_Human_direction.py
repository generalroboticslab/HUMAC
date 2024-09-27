import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np

torch.manual_seed(42)
vicon_offset = [2.66,2.5]

class HideandSeekDataset(Dataset):
   def __init__(self,folder_name,seeker_id,num_seekers,num_of_frame = 5,transform=transforms.ToTensor(),human_control = True,step_ahead = 10):
      self.num_of_frame = num_of_frame
      self.folder_name = folder_name
      self.seeker_id = seeker_id
      self.transform = transform
      self.num_seekers = num_seekers
      self.teammate_position_file_list = []
      self.step_ahead = step_ahead
      self.human_control = human_control

      #check if the dataset matching
      image_folder = self.folder_name+f"/observation/agent_{self.seeker_id}"
      position_file_path = self.folder_name+f"/agent_{self.seeker_id}.txt"
      action_file_path = self.folder_name + f"/next.txt_{self.seeker_id}"
      
      for i in range(num_seekers):
         if i != self.seeker_id:
            self.teammate_position_file_list.append(self.folder_name+f"/agent_{i}.txt")
            
      num_of_images = len(glob.glob(os.path.join(image_folder, '*.png')))
      num_of_actions = 0 
      with open(action_file_path, 'r') as file:
         for line in file:
               # Strip whitespace characters from the line
               if line.strip():
                  num_of_actions += 1 
      
      num_of_positions = 0 
      with open(position_file_path, 'r') as file:
         for line in file:
               # Strip whitespace characters from the line
               if line.strip():
                  num_of_positions += 1 
      try:
         #print(f"folder_name: {folder_name}, num_of_actions: {num_of_actions}, num_of_positions: {num_of_positions}, num_of_images: {num_of_images}")
         assert(num_of_actions >= num_of_positions-1 
               and num_of_positions == num_of_images
               and num_of_actions >= num_of_positions-1 )
      except AssertionError as e:
         print(f"Assertion Error: non-match pairs in {self.folder_name} for Agent{self.seeker_id}")


      self.image_files = [(image_folder + '/' + f) for f in os.listdir(image_folder)]

      self.position_list = []
      with open(position_file_path, 'r') as file:
         for line in file:
            pl = eval(line.strip())
            self.position_list.append(pl[:2])
               
      self.action_list = []
      with open(action_file_path, 'r') as file:
         for line in file:
            al = eval(line.strip())
            self.action_list.append(al[:2])
         
      self.teammate_position_list = []
      for teammate_position_file in self.teammate_position_file_list:
         teammate_position = []
         with open(teammate_position_file, 'r') as file:
            for line in file:
               pl = eval(line.strip())
               teammate_position.append(pl[:2])
         self.teammate_position_list.append(teammate_position)
         
      self.image_files = sorted(self.image_files, key= lambda x: int(x.split('/')[-1].split('_')[0]))
      
      count = 0

      previous_pixel_count = None
      for i, image_file in enumerate(self.image_files):
         image = Image.open(image_file).convert('L')
         image_array = np.array(image)
         black_pixel_count = np.sum(image_array == 0)
         if previous_pixel_count is not None:
            if black_pixel_count >= previous_pixel_count:
               count += 1        
            else:
               break
         previous_pixel_count = black_pixel_count
      self.bad_image = count
      self.time = len(self.image_files)
      self.image_files = self.image_files[self.bad_image:-1]
      
      
      if human_control:
         self.image_files_human = [f for f in self.image_files if 'H' in f.split('/')[-1]]
         self.image_files_human = sorted(self.image_files_human, key= lambda x: int(x.split('/')[-1].split('_')[0]))
      else:
         self.image_files_human = [f for f in self.image_files if 'H' not in f.split('/')[-1]]
         self.image_files_human = sorted(self.image_files_human, key= lambda x: int(x.split('/')[-1].split('_')[0]))

   def __len__(self):
      return len(self.image_files_human)
   
   def is_seeker_fail(self):
      return self.time >= 60 * 5
   
   def have_missing_agent(self):
      bad_count = 0
      for position in self.position_list:
         bad_count += int(sum(position) == 0)
      
      if (bad_count / len(self.position_list)) >= 0.5:
         return True
      else:
         return False
   
   def get(self, index):
      # print(index)
      file_name = self.image_files[index-self.bad_image]
      obs = Image.open(file_name).convert('RGB')

      if self.transform:
         obs = self.transform(obs)
      resize_transform = transforms.Resize((156, 156))
      obs = resize_transform(obs)
                  
      ind = int(file_name.split('/')[-1].split('_')[0])
      
      teammate_position = torch.zeros((2,3)) - 20

      action = torch.zeros((2,1))
      position = torch.zeros((2,1))
      for i in range(len(position)):
         position[i,:] = self.position_list[ind][i]

      if ind + self.step_ahead < len(self.position_list):
         position_after = self.position_list[ind+self.step_ahead]
      else:
         position_after = self.position_list[-1]

      vector = torch.tensor(position_after) - position.squeeze()
      
      if vector.norm() != 0:
         vector = vector / torch.norm(vector)
      action = vector
         
      for i in range(self.num_seekers-1):
         for j in range(len(teammate_position)):
            teammate_position[j,i] = self.teammate_position_list[i][ind][j]
      
      for teammate_index in range(self.num_seekers-1):
         if teammate_position[0,teammate_index] < -19 or teammate_position[1,teammate_index] < -19:
            continue
         y = int((vicon_offset[0] - teammate_position[0,teammate_index])/(2*vicon_offset[0]) * 156)
         x = int((vicon_offset[1] + teammate_position[1,teammate_index])/(2*vicon_offset[1]) * 156)
         pixel_list = []
         for j in range(x-4,x+5):
            for m in range(y-4,y+5):
               if j < 0 or j >= 156 or m < 0 or m >= 156:
                  continue
               if obs[0,j,m] >= 0.6 and obs[1,j,m] <= 0.2 and obs[2,j,m] >= 0.2:
                  pixel_list.append((j,m))
         
         if len(pixel_list) != 0:
            x = int(sum([x[0] for x in pixel_list])/len(pixel_list))
            y = int(sum([x[1] for x in pixel_list])/len(pixel_list))
            
         for j in range(x-2,x+3):
            for m in range(y-2,y+3):   
               if j < 0 or j >= 156 or m < 0 or m >= 156:
                  continue 
               obs[0,j,m] = 1
               obs[1,j,m] = 0
               obs[2,j,m] = 0
            
      ##Make binary mask
      binary_mask = torch.zeros((1,156,156))
      
      y = int((vicon_offset[0] - position[0])/(2*vicon_offset[0]) * 156)
      x = int((vicon_offset[1] + position[1])/(2*vicon_offset[1]) * 156)

      flipped = False
      for j in range(x-2,x+3):
         for m in range(y-2,y+3):
            if j < 0 or j >= 156 or m < 0 or m >= 156:
               continue
            binary_mask[:,j,m] = 1
            flipped = True
      
      
      
      if not flipped:
         if x >= 0 and x < 156 and y >= 0 and y < 156:

            binary_mask[:,x,y] = 1
         else:
            if x < 0:
               x = 0
            if x >= 156:
               x = 155
            if y < 0:
               y = 0
            if y >= 156:
               y = 155
            binary_mask[:,x,y] = 1

      obs = torch.cat((obs,binary_mask),dim = 0)

      return obs,-action,position,teammate_position
   
   def __getitem__(self,ind):
      
      file_name = self.image_files_human[ind]
      index = int(file_name.split('/')[-1].split('_')[0])

      obs_list = []
      num_frames = self.num_of_frame
      last_ind = index
      
      teammate_position_list = []
      for i in range(num_frames):
         prev_index = index - i*5
         if prev_index < 0:
            # print(last_ind)
            obs_i, _, _, teammate_position = self.get(last_ind)
         else:
            # print(prev_index)
            obs_i, _, _, teammate_position = self.get(prev_index)
            last_ind = prev_index
         
         obs_list.append(obs_i)
         teammate_position_list.append(teammate_position)
         
      obs_list.reverse()
      teammate_position_list.reverse()

      obs = torch.cat(obs_list, 0)
      _,action, position, teammate_position = self.get(index)
      teammate_position_list.append(teammate_position)

      return obs, action, position, teammate_position_list


   def plot(self,index):
      observation,action,position, teammate_position_list = self.__getitem__(index)
      
      num_of_frame = self.num_of_frame

      fig, axs = plt.subplots(2, num_of_frame,figsize=(20,10))
      
      img_height, img_width = (2.66*2,2.5*2)

      for i in range(1,num_of_frame+1):
      # Calculate the extent of the image to center it at (0, 0)
         extent = [-img_width / 2, img_width / 2, -img_height / 2, img_height / 2]

         # Display the image with the specified extent
         axs[0,i-1].imshow(observation[(i-1)*4:i*4-1,:,:].permute(1,2,0), extent=extent)
         
         if (i == num_of_frame):
            axs[0,i-1].scatter(0.5*action[0] - position[0], 0.5*action[1] - position[1], c='orange', marker='+', s=40)
            axs[0,i-1].scatter(-position[0], -position[1], c='purple', marker='+', s=40)
         
         print(torch.norm(action))
         # print(position)
         # print(teammate_position[0,teammate_index],teammate_position[1,teammate_index])
         
         
         
         binary_mask = observation[4*i-1,:,:]
         # img_height, img_width = (57.98,58.18)
         extent = [-img_width / 2, img_width / 2, -img_height / 2, img_height / 2]
         axs[1,i-1].imshow(binary_mask, extent=extent, cmap='gray')
         
         # axs[1, i - 1].invert_xaxis()
         # axs[1, i - 1].invert_yaxis()
      plt.savefig(f"example_{index}.png",dpi=500)


