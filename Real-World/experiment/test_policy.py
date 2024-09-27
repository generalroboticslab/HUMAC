import rospy
import json
import threading
from heuristic import heuristic_policy,gamestate_listener, avoid_wall
from utils import vicon_offset,pause_program, green, yellow, red, reset, obstacle_info_dict, initialize_camera, initialize_map, initialize_robot_agent, check_hider_die, get_frame, initialize_cap, start_all_agent, pause_all_agent, resume_all_agent, stop_all_agent
import time
import time
import traceback
from camera import get_seeker_pixel
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch
import os
from datetime import datetime
from model import CustomResNet18, ResNet18_PE_H


now = datetime.now()
seed = f'{now.month}{now.day}{now.hour}{now.minute}{now.second}'
seed = int(seed)

green = "\033[92m"
yellow = "\033[93m"
red = "\033[91m"
reset = "\033[0m"
blue = "\033[94m"

DECISION_FREQUENCY = 0.1
collect_data_decision_frequemcy = 0.2
GAME_TIME = 60
num_of_frames = 3


seeker_info_dict = {"Seeker1":{},"Seeker2":{},"Seeker3":{}}
seeker_action = {"Seeker1":None,"Seeker2":None,"Seeker3":None}
hider_info_dict = {"Hider1":{},"Hider2":{},"Hider3":{}}

live_hider = []


with open('robot_info.json','r') as f:
    info_dict = json.load(f)

def main(cap,seed,IL_Long,FT_Team,folder_name):
    #Initailize the robot
    name_list,robot_agent_list = initialize_robot_agent(info_dict)
    
    num_hider, num_seeker = 0,0
    for name in name_list:
        if name.find("Seeker") != -1:
            num_seeker += 1
        else:
            num_hider += 1
    
    print(f"{green}All Robot Initalized{reset}")
    print(f"{green}Hider Number: {num_hider}, {red}Seeker Number: {num_seeker}{reset}")

    data_path = f"{folder_name}/{num_seeker}Seeker_vs_{num_hider}Hider/seed={seed}"
    
    robot_already_die_list = []
    
    pause_time = 0
        
    try:

        #pause and reset all the agent and map obstacle
        print(f"{yellow}Please put the obstacles in the map{reset}")
        pause_program("1")
        
        # let the camera run from 1s to get rid of the bad frames
        for _ in range(5):
            _ = get_frame(cap)
            time.sleep(0.2)
        
        #take a picture of the empty map
        ref_image = get_frame(cap)     
        resize_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((156, 156)),
        transforms.ToTensor(),
        ])
        ref_image = resize_transform(ref_image)
        
        
        cap.release()
        cap = initialize_cap()
                
        #put the agents in the map            
        print(f"{yellow}Please put the agents in the map{reset}")
        pause_program("2")
        
        f = get_frame(cap)
        
        if os.path.exists(f"{data_path}") == False:
            os.makedirs(f"{data_path}")
        
        save_image(f, f"{data_path}/initial_state.png")
        
        #initalize variables
        map_info = initialize_map(obstacle_info_dict)
        
        hider_already_die_list = []
        acccam_list = []
        
        #add all the accumulated camera in the list
        for name in name_list:
            if name in seeker_info_dict.keys():
                acccam = initialize_camera(156,156, ref_image)
                acccam_list.append(acccam)
        
        memory_list = []
        for name in name_list:
            if name in seeker_info_dict.keys():
                memory_list.append([])
        
        #start the agent
        start_all_agent(robot_agent_list)
        
        print(f"{green}All Robot Started{reset}")

        
        #start the game
        for step in range(int(GAME_TIME/DECISION_FREQUENCY)):
            
            step_start_time = time.time()
            
            #Start the listener for game state
            for name in name_list:
                gamestate_listener(f"/vicon/{name}/{name}",seeker_info_dict,hider_info_dict)
                
            #check if the hider is caught
            hider_already_die_list, new_caught_hider_count = check_hider_die(hider_info_dict, seeker_info_dict,live_hider)

            if new_caught_hider_count > 0:
                pause_start_time = time.time()
                pause_all_agent(robot_agent_list)
                for agent in robot_agent_list:
                    if agent.robot_name in hider_already_die_list:
                        agent.set_light("blue")
                        robot_agent_list.remove(agent)
                        robot_already_die_list.append(agent)
                
                if len(hider_already_die_list) != num_hider:
                    pause_start_time = time.time()
                   
                    pause_program("3")
                    # start_all_agent(robot_agent_list)
                    resume_all_agent(robot_agent_list)
                    pause_time +=  time.time() - pause_start_time
                else:
                    print(f"{red}All hider is caught{reset}")
                    for name in name_list:
                        if name.find("Seeker") != -1:
                            print(f"{red}{name} is caught{reset}")
                    pause_time +=  time.time() - pause_start_time
                    break
                
            
            #get accumulation for camera
            for name in name_list:
                if name in seeker_info_dict.keys():
                    seeker_index = int(name[-1])-1
                    frame = get_frame(cap)
                    
                    acccam = acccam_list[seeker_index]
                    frame,seeker_pixel = get_seeker_pixel(frame,seeker_info_dict[name]['location'],vicon_offset)
                    frame = acccam.add_mask(frame, seeker_pixel)
                    
                    #add an binaty mask along the frame also add red dots on it with teammate location
                    for teammate_name in name_list:
                        if teammate_name in seeker_info_dict.keys() and teammate_name != name:
                            try:
                                teammate_position = seeker_info_dict[teammate_name]['location']
                                y = int((vicon_offset[0] - teammate_position[0])/(2*vicon_offset[0]) * 156)
                                x = int((vicon_offset[1] + teammate_position[1])/(2*vicon_offset[1]) * 156)
                                for j in range(x-3,x+4):
                                    for m in range(y-3,y+4):
                                        if j < 0 or j >= 156 or m < 0 or m >= 156:
                                            continue
                                        
                                        frame[:,0,j,m] = 1
                                        frame[:,1,j,m] = 0
                                        frame[:,2,j,m] = 0
                            except:
                                pass
                    position = seeker_info_dict[name]['location']
                    binary_mask = torch.zeros((1,1,156,156))
      
                    y = int((vicon_offset[0] - position[0])/(2*vicon_offset[0]) * 156)
                    x = int((vicon_offset[1] + position[1])/(2*vicon_offset[1]) * 156)

                    flipped = False
                    for j in range(x-2,x+3):
                        for m in range(y-2,y+3):
                            if j < 0 or j >= 156 or m < 0 or m >= 156:
                                continue
                            binary_mask[:,:,j,m] = 1
                            flipped = True
                    
                    if not flipped:
                        if x >= 0 and x < 156 and y >= 0 and y < 156:

                            binary_mask[:,:,x,y] = 1
                        else:
                            if x < 0:
                                x = 0
                            if x >= 156:
                                x = 155
                            if y < 0:
                                y = 0
                            if y >= 156:
                                y = 155
                            binary_mask[:,:,x,y] = 1
                    
                    frame = torch.cat((frame,binary_mask),dim = 1)
                        
                    memory = memory_list[seeker_index]
                    #save the image for the seeker
                    if (step+1) % (collect_data_decision_frequemcy//DECISION_FREQUENCY) == 0 or step == 0:
                        st = time.time()
                        if len(memory) < (num_of_frames-1)//collect_data_decision_frequemcy+1:
                            memory.append(frame)
                        else:
                            memory.pop(0)
                            memory.append(frame)
                        
                        obs_list = []
                        
                        ind = -1
                        previous_index = -1
                        for i in range(num_of_frames):
                            if i != 0:
                                ind = previous_index - int(1/collect_data_decision_frequemcy)
                            if -ind <= len(memory):
                                obs_list.append(memory[ind])
                                previous_index = ind
                            else:
                                obs_list.append(memory[previous_index])
                        
                        obs_list.reverse()
                        obs = torch.cat(obs_list,dim = 1)
                        
                        if seeker_index != 2:
                            action = IL_Long(obs)
                        else:
                            action = FT_Team(obs)
                        
                        if torch.norm(action) != 0:
                            action = action/torch.norm(action)
                        
                        scale = 0.5
                        
                        action = (-scale*action[0,0].item()+position[0],-scale*action[0,1].item()+position[1])

                        seeker_action[name] = action               
                        seeker_action[name],_ = avoid_wall(seeker_info_dict[name]['location'],seeker_info_dict[name]['direction'],seeker_action[name],map_info)
            
                        
            if step == 0:
                action_dict = {"Seeker1":None,"Seeker2":None,"Seeker3":None,"Hider1":None,"Hider2":None,"Hider3": None}
            else:
                hider_action = heuristic_policy(seeker_info_dict,hider_info_dict,hider_already_die_list,map_info)
            
                action_dict = {**seeker_action,**hider_action}

            threads = []
            for agent in robot_agent_list:
                if agent.robot_name not in hider_already_die_list:
                    thread = threading.Thread(target=agent.set_action,args=(action_dict[agent.robot_name],))
                    threads.append(thread)
                    thread.start()
            if (step+1) % (collect_data_decision_frequemcy//DECISION_FREQUENCY) == 0 or step == 0:
                step_acutal_time = time.time() - step_start_time
            if step_acutal_time < DECISION_FREQUENCY:
                time.sleep(DECISION_FREQUENCY - step_acutal_time)

        pause_all_agent(robot_agent_list)
                
    except:
        print(f"{red}Error Message:{reset}")
        traceback.print_exc()
    
    with open(f"{data_path}/game_result.txt",'a') as f:
        f.write(f"{(step+1)*DECISION_FREQUENCY}\n")
    
    stop_all_agent(robot_agent_list)
    cap.release()
    os._exit(0)

def check_load(model,weight):
    for (_, p1), (_, p2) in zip(model.named_parameters(), weight.items()):
        if torch.equal(p1, p2):
            return True
        break
    return False 

if __name__ == "__main__":
    #load the models
    num_of_frames = 3
    IL_path = "weights/IL_Long_3frame_15step_direction.pth"
    FT_Team_path = "weights/FT_PE_H_3frame_15step_direction.pth"
    
    IL_Long = CustomResNet18(num_of_frames)
    FT_Team = ResNet18_PE_H(num_of_frames)
    Il_state_dict = torch.load(IL_path,map_location=torch.device('cpu'), weights_only=True)["model_state_dict"]
    IL_Long.load_state_dict(Il_state_dict,strict=False)

    FT_Team_dict = torch.load(FT_Team_path,map_location=torch.device('cpu'), weights_only=True)["model_state_dict"]
    FT_Team.load_state_dict(FT_Team_dict)

    if check_load(IL_Long,Il_state_dict):
        print(f"{green}IL_Long model loaded{reset}")
    else:
        print(f"{red}IL_Long model not loaded{reset}")
        
    if check_load(FT_Team,FT_Team_dict):
        print(f"{green}FT_Team model loaded{reset}")
       
    IL_Long.eval()
    FT_Team.eval()
    
    #initialize the node
    rospy.init_node('listener', anonymous=True)
    cap = initialize_cap()
    
    folder_name = "test_clean"
    
    #run the experiment
    main(cap,seed,IL_Long,FT_Team,folder_name)
    
    
    
        

