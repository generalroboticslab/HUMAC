import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

torch.manual_seed(42)

class HideandSeekDataset(Dataset):
   def __init__(self,folder_name,seeker_id,num_seekers,max_seeker_number,num_of_frame = 5,transform=transforms.ToTensor(),step_ahead=5,teammate_only=True):
      self.num_of_frame = num_of_frame
      self.folder_name = folder_name
      self.seeker_id = seeker_id
      self.transform = transform
      self.max_seeker_number = max_seeker_number
      self.num_seekers = num_seekers
      self.step_ahead = step_ahead
      self.teammate_only = teammate_only

      #check if the dataset matching
      image_folder = self.folder_name+f"/observation/agent_{self.seeker_id}"
      position_file_path = self.folder_name+f"/agent_{self.seeker_id}.txt"
      action_file_path = self.folder_name + f"/next.txt_{self.seeker_id}"
      
      num_of_images = len(glob.glob(os.path.join(image_folder, '*.png')))
      num_of_actions = 0 
      with open(action_file_path, 'r') as file:
         for line in file:
               # Strip whitespace characters from the line
               if line.strip():
                  num_of_actions += 1 
      
      #asset the dataset is valid
      num_of_positions = 0 
      with open(position_file_path, 'r') as file:
         for line in file:
               # Strip whitespace characters from the line
               if line.strip():
                  num_of_positions += 1 
      try:
         assert(num_of_actions == num_of_positions 
               and num_of_positions == num_of_images
               and num_of_actions == num_of_positions )
      except AssertionError as e:
         print(f"Assertion Error: non-match pairs in {self.folder_name} for Agent{self.seeker_id}")

      #put all the image path in a list
      self.image_files = [(image_folder + '/' + f) for f in os.listdir(image_folder)]

      #get all the agent position path
      self.position_list = []
      self.position_list_final = []
      with open(position_file_path, 'r') as file:
         for line in file:
            pl = eval(line.strip())
            self.position_list.append([pl[:2]])
            self.position_list_final.append(pl[:2])

      for i in range(self.max_seeker_number):
         if i != self.seeker_id:
            teammate_position_file_path = self.folder_name+f"/agent_{i}.txt"
            if os.path.exists(teammate_position_file_path):
               with open(teammate_position_file_path,'r') as file:
                  for j,line in enumerate(file):
                     self.position_list[j].append(eval(line.strip())[:2])
      
      #get all the agents action 
      self.action_list = []
      with open(action_file_path, 'r') as file:
         for line in file:
            al = eval(line.strip())
            self.action_list.append([al[:2]])

      for i in range(self.max_seeker_number):
         if i != self.seeker_id:
            teammate_action_file_path = self.folder_name+f"/next.txt_{i}"
            if os.path.exists(teammate_action_file_path):
               with open(teammate_action_file_path,'r') as file:
                  for j,line in enumerate(file):
                     self.action_list[j].append(eval(line.strip())[:2])
      
      # sort the action based on the agent's relative position to the current agent
      for j in range(len(self.action_list)):
         index_list = list(range(self.num_seekers))

         index_list = sorted(index_list,key = lambda x:(self.position_list[j][x][0] - self.position_list[j][0][0])**2 + (self.position_list[j][x][1]-self.position_list[j][0][1])**2)

         self.final_action_list = []
         for index in index_list:
            self.final_action_list.append(self.action_list[j][index][0])
            self.final_action_list.append(self.action_list[j][index][1])
         for _ in range(2*(self.max_seeker_number - self.num_seekers)):
            self.final_action_list.append(50)
         self.action_list[j] = self.final_action_list
   
      
      #sort the images files based on the step number
      self.image_files = sorted(self.image_files, key= lambda x: int(x.split('/')[-1][:-4]))
      self.image_files = self.image_files[:self.num_of_frame-self.step_ahead+1]

   def __len__(self):
      return (len(self.image_files))
   
   def is_seeker_fail(self):
      return len(self) == 120
   
   def have_missing_agent(self):
      bad_count = 0
      for position in self.position_list_final:
         bad_count += int(sum(position) == 0)
      
      if (bad_count / len(self.position_list_final)) >= 0.5:
         return True
      else:
         return False
   
   def get(self, index):
      file_name = self.image_files[index]
      obs = Image.open(file_name).convert('RGB')

      if self.transform:
         obs = self.transform(obs)
      ind = int(file_name.split('/')[-1][:-4])
      ind1 = ind + self.step_ahead-1

      if self.teammate_only:
         action = torch.zeros((2*self.max_seeker_number-2,1))
         for i in range(len(action)):
            action[i,:] = self.action_list[ind1][i+2]
      else:
         action = torch.zeros((2*self.max_seeker_number,1))
         for i in range(len(action)):
            action[i,:] = self.action_list[ind1][i]
      
      position = torch.zeros((2,1))
      for i in range(len(position)):
         position[i,:] = self.position_list_final[ind][i]
      
      ##Make binary mask
      binary_mask = torch.zeros((1,156,156))
      
      x = int((-position[1]+28.99)/57.98*155)
      y = int((position[0]+29.09)/58.18*155)

      flipped = False
      for j in range(x-3,x+4):
         for m in range(y-3,y+4):
            if j < 0 or j >= 156 or m < 0 or m >= 156:
               continue
            if obs[0,j,m] >= 0.85 and obs[1,j,m] <= 0.22 and obs[2,j,m] <= 0.22:
               binary_mask[:,j,m] = 1
               flipped = True
      
      if not flipped:
         binary_mask[:,x,y] = 1

      obs = torch.cat((obs,binary_mask),dim = 0)

      return obs,action,position
   
   def __getitem__(self,ind):
      
      file_name = self.image_files[ind]
      index = int(file_name.split('/')[-1][:-4])

      obs_list = []
      num_frames = self.num_of_frame
      last_ind = ind
      for i in range(num_frames):
         prev_index = index - i*5
         if prev_index < 0:
            obs_i, _, _ = self.get(last_ind)
         else:
            obs_i, _, _ = self.get(ind - i*5)
            last_ind = ind - i*5
         obs_list.append(obs_i)

      obs_list.reverse()

      obs = torch.cat(obs_list, 0)
      _,action, position = self.get(ind)

      return obs, action, position


   def plot(self,index):
      observation,action,position = self.__getitem__(index)
      
      num_of_frame = self.num_of_frame

      fig, axs = plt.subplots(2, num_of_frame,figsize=(20,10))
      
      img_height, img_width = (57.98,58.18)

      for i in range(1,num_of_frame+1):
      # Calculate the extent of the image to center it at (0, 0)
         extent = [-img_width / 2, img_width / 2, -img_height / 2, img_height / 2]

         # Display the image with the specified extent
         axs[0,i-1].imshow(observation[(i-1)*4:i*4-1,:,:].permute(1,2,0), extent=extent)

         if (i == num_of_frame):
            for j in range(self.max_seeker_number):
               try:
                  if action[2*j] != 50:
                     alpha = 1 - j / self.max_seeker_number
                     axs[0,i-1].scatter(action[2*j], action[2*j+1], c='orange', marker='+', s=20,alpha = alpha)
               except:
                  pass
            axs[0,i-1].scatter(position[0], position[1], c='black', marker='+', s=10)
         
         binary_mask = observation[4*i-1,:,:]
         img_height, img_width = (57.98,58.18)
         extent = [-img_width / 2, img_width / 2, -img_height / 2, img_height / 2]
         axs[1,i-1].imshow(binary_mask, extent=extent, cmap='gray')
      plt.savefig("example.jpg",dpi = 400)


