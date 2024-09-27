import rospy
import json
import threading
from heuristic import heuristic_policy,gamestate_listener
from utils import vicon_offset,pause_program, green, yellow, red, reset, obstacle_info_dict, initialize_camera, initialize_map, initialize_robot_agent, check_hider_die, get_frame, initialize_cap, start_all_agent, pause_all_agent, resume_all_agent, stop_all_agent
import time
import traceback
from camera import  get_seeker_pixel
from torchvision.utils import save_image
import torchvision.transforms as transforms
import os
from datetime import datetime
import sys

now = datetime.now()
seed = f'{now.month}{now.day}{now.hour}{now.minute}{now.second}'
seed = int(seed)

DECISION_FREQUENCY = 0.1
collect_data_decision_frequemcy = 0.2
GAME_TIME = 60 + 3 #3 additional seconds for the data collection

seeker_info_dict = {"Seeker1":{},"Seeker2":{},"Seeker3":{}}
hider_info_dict = {"Hider1":{},"Hider2":{},"Hider3":{}}
live_hider = []

with open('robot_info.json','r') as f:
    info_dict = json.load(f)

def main(cap,seed,folder_name):
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
        
    try:

        #pause and reset all the agent and map obstacle
        print(f"{yellow}Please put the obstacles in the map{reset}")
        pause_program("1")
        
        # let the camera run from 1s to get rid of the bad frames
        for _ in range(5):
            frame = get_frame(cap)
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
                
                pause_all_agent(robot_agent_list)
                for agent in robot_agent_list:
                    if agent.robot_name in hider_already_die_list:
                        agent.set_light("blue")
                        robot_agent_list.remove(agent)
                        robot_already_die_list.append(agent)
                
                if len(hider_already_die_list) != num_hider:
                    
                    pause_program("3")
                    resume_all_agent(robot_agent_list)
            
                else:
                    print(f"{red}All hider is caught{reset}")
                    for name in name_list:
                        if name.find("Seeker") != -1:
                            print(f"{red}{name} is caught{reset}")
                            with open(f"{data_path}/next.txt_{seeker_index}",'a') as f:
                                f.write(f"{list(seeker_info_dict[name]['location'])}\n")

                    break

            #get accumulation for camera
            for name in name_list:
                if name in seeker_info_dict.keys():
                    seeker_index = int(name[-1])-1
                    if os.path.exists(f"{data_path}/observation/agent_{seeker_index}") == False:
                        os.makedirs(f"{data_path}/observation/agent_{seeker_index}")
                    
                    frame = get_frame(cap)
                    
                    acccam = acccam_list[seeker_index]
                    frame,seeker_pixel = get_seeker_pixel(frame,seeker_info_dict[name]['location'],vicon_offset)
                    frame = acccam.add_mask(frame, seeker_pixel)
                        

                    #save the image for the seeker
                    if (step+1) % (collect_data_decision_frequemcy//DECISION_FREQUENCY) == 0 or step == 0:
                        save_image(frame, f"{data_path}/observation/agent_{seeker_index}/{int((step+1)//(collect_data_decision_frequemcy//DECISION_FREQUENCY))}.png")
                            
                        #write the location of the seeker to a file
                        with open(f"{data_path}/agent_{seeker_index}.txt",'a') as f:
                            f.write(f"{list(seeker_info_dict[name]['location'])}\n")
                        
                        if step != 0:
                            with open(f"{data_path}/next.txt_{seeker_index}",'a') as f:
                                f.write(f"{list(seeker_info_dict[name]['location'])}\n")

            #calculate the action for the agent
            if step == 0:
                action_dict = {"Seeker1":None,"Seeker2":None,"Seeker3":None,"Hider1":None,"Hider2":None,"Hider3": None}
            else:
                action_dict = heuristic_policy(seeker_info_dict,hider_info_dict,hider_already_die_list,map_info)
   
            #excute actions for the agent
            threads = []
            for agent in robot_agent_list:
                if agent.robot_name not in hider_already_die_list:
                    thread = threading.Thread(target=agent.set_action,args=(action_dict[agent.robot_name],))
                    threads.append(thread)
                    thread.start()
            
            step_acutal_time = time.time() - step_start_time
            
            if step_acutal_time < DECISION_FREQUENCY:
                time.sleep(DECISION_FREQUENCY - step_acutal_time)

            #stop the robot
        pause_all_agent(robot_agent_list)
                
    except:
        print(f"{red}Error Message:{reset}")
        traceback.print_exc()
    
    with open(f"{data_path}/game_result.txt",'a') as f:
        f.write(f"{(step+1)*DECISION_FREQUENCY}\n")
    
    stop_all_agent(robot_agent_list)
    cap.release()
    os._exit(0)

if __name__ == "__main__":
    rospy.init_node('listener', anonymous=True)
    cap = initialize_cap()
    folder_name = "data_clean"
    
    main(cap,seed,folder_name)
    
    
    
        

