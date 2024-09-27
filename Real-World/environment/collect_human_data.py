from robomaster import conn
from geometry_msgs.msg import TransformStamped
import rospy
import json
from robot_utils import RobotAgent
import threading
from heuristic import heuristic_policy,gamestate_listener,get_distance,find_box_corners,quaternion_to_dictional_vector
import time
import traceback
from functools import partial
from camera import AccumulatedCamera, process_image
import cv2
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch
import os
from PIL import Image
from datetime import datetime
from streaming import VideoStreamUI,image_size
from PyQt5.QtWidgets import QApplication
import sys
from utils import pause_program, green, yellow, red, reset, obstacle_info_dict, initialize_camera, initialize_map, initialize_robot_agent, check_hider_die, initialize_cap, start_all_agent, pause_all_agent, resume_all_agent, stop_all_agent

now = datetime.now()
seed = f'{now.month}{now.day}{now.hour}{now.minute}{now.second}'
seed = int(seed)
DECISION_FREQUENCY = 0.1
collect_data_decision_frequemcy = 0.2
GAME_TIME = 60

saved_image_size = 156


seeker_info_dict = {"Seeker1":{},"Seeker2":{},"Seeker3":{}}
hider_info_dict = {"Hider1":{},"Hider2":{},"Hider3":{}}


live_hider = []


def initialize_camera(width,height, ref_img):
    # Open the default camera (usually the webcam)
    acccam = AccumulatedCamera(width, height, ref_img, mask_size=int(3/5*image_size))    
    return acccam

def get_frame(cap,save=False):
    ret, frame = cap.read()
    if save:
        image = Image.fromarray(frame)

        # Save the image
        image.save('image_get.png')
    if not ret:
        print("Error: Could not read frame.")
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0) / 255
    frame = frame.numpy().squeeze().transpose(1,2,0)
    frame = process_image(frame)
    frame = torch.tensor(frame).permute(2, 0, 1)
    #reszie the image
    frame = transforms.Resize((image_size,image_size))(frame)
    return frame

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
        ros_thread = threading.Thread(target=rospy.init_node, args=('listener',), kwargs={'anonymous': True})
        ros_thread.start()

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
        transforms.ToTensor(),
        ])
        print(ref_image.shape)
        ref_image = resize_transform(ref_image)
        
        #reset the camera
        cap.release()
        cap = initialize_cap()
        
        #start UI
        print(f"{yellow}Please start the UI and the game{reset}")
        pause_program("2")
        
        #define accumulative camera list
        acccam_list = []
        for name in name_list:
            acccam = initialize_camera(image_size,image_size, ref_image)
            acccam_list.append(acccam)

        #record the initial state of the map
        f = get_frame(cap)
        if os.path.exists(f"{data_path}") == False:
            os.makedirs(f"{data_path}")
        save_image(f, f"{data_path}/initial_state.png")        
        
        
        app = QApplication(sys.argv)
        window = VideoStreamUI(cap,num_seeker, acccam_list)
        
        #initalize variables
        map_info = initialize_map(obstacle_info_dict)
        
        #start the agent
        start_all_agent(robot_agent_list)
        print(f"{green}All Robot Started{reset}")
        
        #record the start time
        start_time = time.time()
        
        main_loop_thread = threading.Thread(target=main_loop, args=(window,
                                                                    name_list,
                                                                    robot_agent_list,
                                                                    seeker_info_dict,
                                                                    hider_info_dict,
                                                                    map_info,
                                                                    data_path,
                                                                    num_hider,
                                                                    robot_already_die_list,
                                                                    start_time))
        main_loop_thread.start()
        
        window.show()
        
        sys.exit(app.exec_())
                
    except:
        print(f"{red}Error Message:{reset}")
        traceback.print_exc()
    

    
    stop_all_agent(robot_agent_list)

def main_loop(window,
              name_list,
              robot_agent_list,
              seeker_info_dict,
              hider_info_dict,
              map_info,
              data_path,
              num_hider,
              robot_already_die_list,
              start_time):
    try:
        for step in range(int(GAME_TIME/DECISION_FREQUENCY)):            
            step_start_time = time.time()
            action_dict = {"Seeker1":None,"Seeker2":None,"Seeker3":None,"Hider1":None,"Hider2":None,"Hider3": None}
            window.update_seeker_location(seeker_info_dict,action_dict)
            
            #Start the listener for game state
            game_thread_list = []
            for name in name_list:
                # Create a thread for each name, targeting the gamestate_listener function
                thread = threading.Thread(target=gamestate_listener, args=(f"/vicon/{name}/{name}", seeker_info_dict, hider_info_dict))
                game_thread_list.append(thread)
                thread.start()
                
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

                    break
            
            
            
            # get accumulation for camera
            for name in name_list:
                if name in seeker_info_dict.keys():
                    seeker_index = int(name[-1])-1
                    if os.path.exists(f"{data_path}/observation/agent_{seeker_index}") == False:
                        os.makedirs(f"{data_path}/observation/agent_{seeker_index}")
                    
                    frame = window.frame_list[seeker_index]
                    frame = transforms.Resize((saved_image_size,saved_image_size))(frame)
                    if window.human_control[seeker_index]:
                        is_human_control = "H"
                    else:
                        is_human_control = ""
                    # save the image for the seeker
                    if (step+1) % (collect_data_decision_frequemcy//DECISION_FREQUENCY) == 0 or step == 0:
                        save_image(frame, f"{data_path}/observation/agent_{seeker_index}/{int((step+1)//(collect_data_decision_frequemcy//DECISION_FREQUENCY))}_{is_human_control}.png")
                            
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

            for is_human_control,human_action in zip(window.human_control,window.human_action):
                if is_human_control:
                    agent_name = f"Seeker{window.human_action.index(human_action)+1}"
                    action_dict[agent_name] = human_action
            
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

        
        with open(f"{data_path}/game_result.txt",'a') as f:
            f.write(f"{(step+1)*DECISION_FREQUENCY}\n")
        #stop the robot
        pause_all_agent(robot_agent_list)
        os._exit(0)
    except:
        print(f"{red}Error Message:{reset}")
        traceback.print_exc()
    stop_all_agent(robot_agent_list)

if __name__ == "__main__":
    folder_name = "data_clean_human"
    cap = initialize_cap()
    main(cap,seed,folder_name)
    
    
    
        

