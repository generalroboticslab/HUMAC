# from Hide_and_Seek_Dataset_real import HideandSeekDataset
import os
# from torch.utils.data import ConcatDataset
# from torch.utils.data import  DataLoader
# from model import CustomResNet18
# import torch
# from matplotlib import pyplot as plt
root_folder = "test"


for folder in os.scandir(root_folder):
    for policy_folder in os.scandir(folder):
    
        total_fail = 0
        total_episode = 0

        base_folder = os.path.join(root_folder, folder.name, policy_folder.name)
        num_seekers = int(str(folder.name)[0])

        # if num_seekers != 2:
        #     continue
        total_success = 0
        total_episode = 0
        for seed_num in os.scandir(base_folder):
            total_episode += 1
            resultPath = os.path.join(base_folder,seed_num.name,"game_result.txt")
            with open(resultPath, "r") as f:
                time = float(f.readline())
                if time < 60:
                    total_success += 1
        print(f"{folder.name} {policy_folder.name} success rate: {total_success/total_episode}, total episode: {total_episode}")            