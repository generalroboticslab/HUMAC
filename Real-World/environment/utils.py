from robomaster import conn
from geometry_msgs.msg import TransformStamped
import rospy
from robot_utils import RobotAgent
import threading
from heuristic import get_distance,find_box_corners,quaternion_to_dictional_vector
import time
from functools import partial
from camera import AccumulatedCamera, process_image
import cv2
import torch
from PIL import Image

green = "\033[92m"
yellow = "\033[93m"
red = "\033[91m"
reset = "\033[0m"
blue = "\033[94m"

image_size = 156

vicon_offset = [2.66,2.5]


global obstacle_info_dict
obstacle_info_dict = {"Square":(0.26*2,0.21*2),"Rectangle":(0.42*2,0.13*2),"Cross":(0.42*2,0.13*2),"Lshape":(0.42*2,0.13*2)}
global Obstacle_positions
Obstacle_positions = [None,None,None]
global Obstacle_directions
Obstacle_directions = [None,None,None]
Obstacale_name = ["Rectangle","Cross"]

live_hider = []

def pause_program(input_string):
    '''
    pause the program until the user give the input string
    
    Parameters:
    - input_string (str): The string that the user needs to input to continue
    '''
    while True:
        user_input = input(f"{blue}Make sure you set up all the obstacle and agent. Type '{input_string}' to continue: {reset}").strip().lower()
        if user_input == input_string:
            break

def listener(path, callback):
    '''
    listener to listen to the topic
    '''
    rospy.Subscriber(path, TransformStamped, partial(callback,agent_name=path.split("/")[-1]))

def callback(data,agent_name):    
    '''
    callback function to get the robot's position and orientation
    '''
    ind = Obstacale_name.index(agent_name)
    
    Obstacle_positions[ind] = (data.transform.translation.x, data.transform.translation.y)
    Obstacle_directions[ind] = quaternion_to_dictional_vector((data.transform.rotation.x, data.transform.rotation.y, data.transform.rotation.z, data.transform.rotation.w))
       
def initialize_cap():
    '''
    Initialize the camera
    '''
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture device.")
        exit()
    return cap

def initialize_camera(width,height, ref_img):
    '''
    Initialize the accamulation camera
    '''
    # Open the default camera (usually the webcam)
    acccam = AccumulatedCamera(width, height, ref_img, mask_size=int(3/5*image_size))    
    return acccam

def get_frame(cap,save=False):
    '''
    Get the frame from the camera
    '''
    
    ret, frame = cap.read()
    if save:
        image = Image.fromarray(frame)

        # Save the image
        image.save('image_get.png')
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if not ret:
        print("Error: Could not read frame.")
        return None
    frame = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0) / 255
    frame = frame.numpy().squeeze().transpose(1,2,0)
    frame = process_image(frame)
    frame = torch.tensor(frame).permute(2, 0, 1)
    return frame

def initialize_map(obstacle_info_dict):
    '''
    Initialize the map with the obstacle
    '''
    
    four_corners_list = []
    rospy.init_node('listener', anonymous=True)
    for name in Obstacale_name:
        path = f"/vicon/{name}/{name}"
        listener(path, callback)
        time.sleep(0.02)
    
    for obstacle_name,obstacle_pos,obstacle_dir in zip(Obstacale_name,Obstacle_positions,Obstacle_directions):
        if obstacle_dir is not None:
            length,width = obstacle_info_dict[obstacle_name]
            four_corners = find_box_corners(obstacle_pos[0],
                                            obstacle_pos[1],
                                            obstacle_dir[0],
                                            obstacle_dir[1],
                                            length,
                                            width)
            four_corners_list.append(four_corners)
            
            if obstacle_name == "Cross":
                
                length,width = obstacle_info_dict[obstacle_name]
                four_corners = find_box_corners(obstacle_pos[0],
                                                obstacle_pos[1],
                                                -obstacle_dir[1],
                                                obstacle_dir[0],
                                                length,
                                                width)
                four_corners_list.append(four_corners)
            
            if obstacle_name == "Lshape":
                length,width = obstacle_info_dict[obstacle_name]
                four_corners = find_box_corners(obstacle_pos[0] - (length-width)/2 ,
                                                obstacle_pos[1] - (length-width)/2,
                                                -obstacle_dir[1],
                                                obstacle_dir[0],
                                                length,
                                                width)
                four_corners_list.append(four_corners)
    print(f"{green} Map Initialized{reset}")
    return four_corners_list

def initialize_robot_agent(info_dict):
    '''
    Initialize the robot agent
    '''
    ip_list = conn.scan_robot_ip_list(timeout=1)
    
    robot_agent_list = []
    name_list = []
        
    for ip in ip_list:
        sn,name = info_dict[ip][0],info_dict[ip][1]
        robot_agent = RobotAgent(sn,name)

        if name.find("Seeker") != -1:
            robot_agent.set_light("red")
        else:
            robot_agent.set_light("green")

        robot_agent_list.append(robot_agent)
        name_list.append(name)
    
    return name_list,robot_agent_list

def start_all_agent(robot_agent_list):
    '''
    start all the robot agent
    '''
    threads = []
    for agent in robot_agent_list:
        thread = threading.Thread(target=agent.start_robot)
        threads.append(thread)
        thread.start()
    
def stop_all_agent(robot_agent_list):
    '''
    stop all the robot agent
    '''
    threads = []
    for agent in robot_agent_list:
        thread = threading.Thread(target=agent.pause)
        threads.append(thread)
        thread.start()

    
    threads = []
    for agent in robot_agent_list:
        thread = threading.Thread(target=shutdown_agent, args=(agent,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

def resume_all_agent(robot_agent_list):
    '''
    resume all the robot agent
    '''
    threads = []
    for agent in robot_agent_list:
        thread = threading.Thread(target=agent.resume)
        threads.append(thread)
        thread.start()

def pause_all_agent(robot_agent_list):
    '''
    pause all the robot agent
    '''
    
    threads = []
    for agent in robot_agent_list:
        thread = threading.Thread(target=agent.pause)
        threads.append(thread)
        thread.start()

def check_hider_die(hider_info_dict, seeker_info_dict,hider_already_die_list):
    '''
    check if any hider is caught by the seeker
    '''

    inital_hider_caught_count = len(hider_already_die_list)
    for hider_name in hider_info_dict.keys():
        if hider_name not in hider_already_die_list:
            for seeker_name in seeker_info_dict.keys():
                try:
                    hider_location = hider_info_dict[hider_name]["location"]
                    seeker_location = seeker_info_dict[seeker_name]["location"]
                    if get_distance(hider_location,seeker_location) < 0.4:
                        print(f"{red}{hider_name} is caught by {seeker_name}{reset}")

                        if hider_name not in hider_already_die_list:
                            hider_already_die_list.append(hider_name)
                    
                except:
                    pass
    return hider_already_die_list,len(hider_already_die_list) - inital_hider_caught_count
    
def change_normal_scale(self_location,target):
    x,y = target
    x = x - self_location[0]
    y = y - self_location[1]
    return x,y

def shutdown_agent(agent):
    '''
    shutdown the agent
    '''
    agent.get_battery_info()
    agent.shutdown()