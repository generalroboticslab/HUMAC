import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

torch.manual_seed(42)
vicon_offset = [2.66,2.5]

class HideandSeekDataset(Dataset):
   def __init__(self,folder_name,seeker_id,num_seekers,num_of_frame = 5,transform=transforms.ToTensor(),step_ahead = 10):
      self.num_of_frame = num_of_frame
      self.folder_name = folder_name
      self.seeker_id = seeker_id
      self.transform = transform
      self.num_seekers = num_seekers
      self.teammate_position_file_list = []
      self.step_ahead = step_ahead

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
         
      # self.action_list.append(self.action_list[-1])
      self.image_files = sorted(self.image_files, key= lambda x: int(x.split('/')[-1][:-4]))
      #get rid of the last image in the list
      self.image_files = self.image_files[:-1]

   def __len__(self):
      return len(self.image_files)
   
   def is_seeker_fail(self):
      return len(self) >= 60 * 5
   
   def have_missing_agent(self):
      bad_count = 0
      for position in self.position_list:
         bad_count += int(sum(position) == 0)
      
      if (bad_count / len(self.position_list)) >= 0.5:
         return True
      else:
         return False
   
   def get(self, index):
      file_name = self.image_files[index]
      obs = Image.open(file_name).convert('RGB')

      if self.transform:
         obs = self.transform(obs)
      
               
      ind = int(file_name.split('/')[-1][:-4])
      
      teammate_position = torch.zeros((2,3)) - 20

      action = torch.zeros((3,1))
      for i in range(len(action)-1):
         if ind + self.step_ahead >= len(self.position_list):
            action[i,:] = self.action_list[-1][i]
         else:
            action[i,:] = self.position_list[ind+self.step_ahead][i]

      position = torch.zeros((2,1))
      for i in range(len(position)):
         position[i,:] = self.position_list[ind][i]
   
      off_set = 10
      if ind - off_set >= 0:
         pre_5_position = self.position_list[ind-off_set]
      else:
         pre_5_position = self.position_list[0]
      
      if ind + off_set < len(self.position_list):
         post_5_position = self.position_list[ind+off_set]
      else:
         post_5_position = self.position_list[-1]
         
      vector1 = torch.tensor([position[0] - pre_5_position[0], position[1] - pre_5_position[1]])
      vector2 = torch.tensor([post_5_position[0] - position[0], post_5_position[1] - position[1]])
      
      
      if torch.dot(vector1.squeeze(),vector2.squeeze()) < 0:
         
         z = vector1[0] * vector2[1] - vector1[1] * vector2[0]
         if z < 0:
            action[-1,:] = -1
         elif z > 0:
            action[-1,:] = -2
         else:
            action[-1,:] = 0
      else:
         action[-1,:] = 0
         
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
      
      file_name = self.image_files[ind]
      index = int(file_name.split('/')[-1][:-4])

      obs_list = []
      num_frames = self.num_of_frame
      last_ind = ind
      teammate_position_list = []
      for i in range(num_frames):
         prev_index = index - i*5
         if prev_index < 0:
            # print(last_ind)
            obs_i, _, _, teammate_position = self.get(last_ind)
         else:
            # print(ind-5*i)
            obs_i, _, _, teammate_position = self.get(ind - i*5)
            last_ind = ind - i*5
         
         obs_list.append(obs_i)
         teammate_position_list.append(teammate_position)
         
      obs_list.reverse()
      teammate_position_list.reverse()

      obs = torch.cat(obs_list, 0)
      _,action, position, teammate_position = self.get(ind)
      teammate_position_list.append(teammate_position)

      return obs, action, position, teammate_position_list


   def plot(self,index):
      observation,action,position, teammate_position_list = self.__getitem__(index)
      
      num_of_frame = self.num_of_frame

      fig, axs = plt.subplots(2, num_of_frame,figsize=(20,10))
      
      img_height, img_width = (2.66*2,2.5*2)

      for i,teammate_position in zip(range(1,num_of_frame+1),teammate_position_list):
      # Calculate the extent of the image to center it at (0, 0)
         extent = [-img_width / 2, img_width / 2, -img_height / 2, img_height / 2]

         # Display the image with the specified extent
         axs[0,i-1].imshow(observation[(i-1)*4:i*4-1,:,:].permute(1,2,0), extent=extent)
         
         # axs[0, i - 1].invert_xaxis()
         # axs[0, i - 1].invert_yaxis()

         if (i == num_of_frame):
            axs[0,i-1].scatter(action[0], action[1], c='orange', marker='+', s=40)
            # axs[0,i-1].scatter(-position[0], -position[1], c='blue', marker='o', s=30)
         
         # print(position)
         # print(teammate_position[0,teammate_index],teammate_position[1,teammate_index])
         
         
         
         binary_mask = observation[4*i-1,:,:]
         # img_height, img_width = (57.98,58.18)
         extent = [-img_width / 2, img_width / 2, -img_height / 2, img_height / 2]
         axs[1,i-1].imshow(binary_mask, extent=extent, cmap='gray')
         
         # axs[1, i - 1].invert_xaxis()
         # axs[1, i - 1].invert_yaxis()
      plt.savefig("example_1.png",dpi=500)

