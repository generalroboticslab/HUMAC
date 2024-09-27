import math
from robomaster import robot
import rospy
from geometry_msgs.msg import TransformStamped
import threading
import time
import tf.transformations as tf_trans
import numpy as np

green = "\033[92m"
yellow = "\033[93m"
red = "\033[91m"
reset = "\033[0m"
blue = "\033[94m"

def calculate_turning_angle(x,y):
    '''
    Calculate the turning angle of the robot towards a target based on its coordinates.

    The function computes the turning angle required for a robot to face a target located at coordinates (x, y) relative to its current position, assuming the robot is initially facing along the positive x-axis. The angle is returned in degrees and is measured counterclockwise, with negative values indicating a clockwise rotation.

    Parameters:
    - x (float): The x-coordinate of the target position.
    - y (float): The y-coordinate of the target position.

    Returns:
    - float: The turning angle in degrees. The value is positive for counterclockwise rotations and negative for clockwise rotations.
    '''
    if x == 0 and y != 0:
        if y > 0:
            return -90
        else:
            return 90
    
    elif x != 0 and y != 0:
        tan_value = abs(y/x)
        radians = math.atan(tan_value)
        degrees = math.degrees(radians)
        if x > 0 and y > 0:
            return -degrees
        
        elif x > 0 and y < 0: 
            return degrees
            
        elif x < 0 and y > 0:
            return -180+degrees
        else:
            return 180-degrees
    
    else:
        if x < 0:
            return 180


class RobotAgent:
    def __init__(self,sn,name):
        self.robot = robot.Robot()
        
        try:
            self.robot.initialize(conn_type="sta",sn=sn)
        except:
            print(f"{red}Failed to initialize the robot with sn: {reset}",sn)    
        
        self.robot_name = name
        self.speed = 0
        
        if self.robot_name[0] == "S":
            self.isHider = False
        else:
            self.isHider = True
        
        self.location = None
        self.orientation = None
        self.listen_to_action = True
        
        self.current_action = self.location

        # Start the listener in a separate thread
        rospy.init_node('listener', anonymous=True)
        self.listener_thread = threading.Thread(target=self.start_listener, args=(f"vicon/{name}/{name}",))
        self.listener_thread.start()
        
        self.command_thread = threading.Thread(target=self.command_listener)
        self.command_thread.start()
    
    def set_light(self,color):
        '''
        Set the color of the robot's light
        
        Parameters:
        - color (str): The color of the light. The possible values are "green", "red", and "blue".
        "green is the hider", "red is the seeker", and "blue mean dead".
        '''
        if color == "green":
            color = (0,255,0)
        elif color == "red":
            color = (255,0,0)
        elif color == "blue":
            color = (0,0,255)
        else:
            return
        self.robot.led.set_led(comp="all", r=color[0], g=color[1], b=color[2], effect="on")
        
    def start_listener(self, path):
        '''
        Start the listener to listen to the robot's position and orientation
        
        Parameters:
        - path (str): The path to the topic to listen to
        '''
        rospy.Subscriber(path, TransformStamped, self.callback)
        
    def callback(self, data):
        '''
        Callback function to get the robot's position and orientation
        
        Parameters:
        - data (TransformStamped): The data containing the robot's position and orientation
        '''
        self.location = (data.transform.translation.x, data.transform.translation.y, 0)
        self.orientation = (0, 0, data.transform.rotation.z, data.transform.rotation.w)
    
    def start_robot(self):
        '''
        Start the robot
        '''
        if self.isHider:
            self.speed = 130
        else:
            self.speed = 65
    
    def stop(self):
        '''
        Stop the robot
        '''
        
        self.robot.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0, timeout=None)
        self.w1_speed = 0
        self.w2_speed = 0
        self.w3_speed = 0
        self.w4_speed = 0
    
    def pause(self):
        '''
        pause teh robot
        '''
        self.listen_to_action = False
        
    def command_listener(self):
        '''
        Callback function to listen to the commands
        '''
        while self.listen_to_action:
            if self.current_action is not None:
                self.move(self.current_action[0],self.current_action[1])
            else:
                self.reset_speed()
            
            self.robot.chassis.drive_wheels(w1=self.w1_speed, w2=self.w2_speed, w3=self.w3_speed, w4=self.w4_speed, timeout=None)
        self.stop()
    
    def resume(self):
        '''
        Resume the robot
        '''
        self.w1_speed = self.speed
        self.w2_speed = self.speed
        self.w3_speed = self.speed
        self.w4_speed = self.speed
        self.robot.chassis.drive_wheels(w1=self.w1_speed, w2=self.w2_speed, w3=self.w3_speed, w4=self.w4_speed, timeout=None)
        self.listen_to_action = True
        
        self.command_thread1 = threading.Thread(target=self.command_listener)
        self.command_thread1.start()
        
        
    def move(self,x,y):
        '''
        Move the robot to a certain position
        
        Parameters:
        - x (float): The x-coordinate of the target position.
        - y (float): The y-coordinate of the target position.
        '''
        
        while self.location is None and self.orientation is None:
            pass
        
        x,y,_ = self.translate_to_agent_coordinate((x,y,0))
        y = -y # y is inverted in the agent coordinate system
        
        if x == 0 and y == 0:
            return
        
        turn_angle = calculate_turning_angle(x,y)
        
        if turn_angle > 0 and turn_angle <= 60:
            self.w1_speed = self.speed
            self.w2_speed = 0.6*self.speed
            self.w3_speed = 0.6*self.speed
            self.w4_speed = self.speed
            
        elif turn_angle >= -60 and turn_angle < 0:
            self.w1_speed = 0.6*self.speed
            self.w2_speed = self.speed
            self.w3_speed = self.speed
            self.w4_speed = 0.6*self.speed
            
        elif turn_angle > 60:
            self.w1_speed = self.speed
            self.w2_speed = -self.speed
            self.w3_speed = -self.speed
            self.w4_speed = self.speed
        else:
            self.w1_speed = -self.speed
            self.w2_speed = self.speed
            self.w3_speed = self.speed
            self.w4_speed = -self.speed


    def turn(self,angle):
        '''
        Turn the robot by a certain angle
        
        parameters:
        - angle (float): The angle by which the robot should turn. Positive values indicate counterclockwise rotations, while negative values indicate clockwise rotations
        '''
        self.robot.chassis.move(x=0, y=0, z=angle, z_speed=360).wait_for_completed()
        
    def set_action(self,action):
        '''
        Set the current action of the robot
        
        action (tuple): The action to be performed by the robot. The action is represented as a tuple (x, y), where x and y are the x and y coordinates of the target position, respectively.
        '''
        self.current_action = action
        
    def reset_speed(self):
        '''
        Reset the speed of the robot
        '''
        self.w1_speed = self.speed
        self.w2_speed = self.speed
        self.w3_speed = self.speed
        self.w4_speed = self.speed
    
    def shutdown(self):
        '''
        Shutdown the robot
        '''
        self.listen_to_action = False
        self.robot.close()
        if self.listener_thread.is_alive():
            rospy.signal_shutdown('Shutting down')
            print(f"{self.robot_name} is shutdown")
            self.listener_thread.join()
        
    def get_battery_info(self):
        '''
        Get the battery information of the robot
        '''
        self.robot.battery.sub_battery_info(10, self.sub_info_handler, self.robot)
        time.sleep(0.15)
        self.robot.battery.unsub_battery_info()
    
    def sub_info_handler(self,batter_info):
        '''
        Callback function to get the battery information
        
        parameters:
        - batter_info (int): The battery information
        '''
        percent = batter_info
        print(f"{self.robot_name} Battery: {yellow}{percent}%{reset}")
    
    
    def translate_to_agent_coordinate(self,world_point):
        '''
        Translate a point from the world coordinate system to the robot coordinate system
        
        world_point (tuple): The point in the world coordinate system
        '''
        rotation_matrix = tf_trans.quaternion_matrix(self.orientation)
        rotation_matrix[:3, 3] = self.location
        homogeneous_point = np.array([*world_point, 1.0])
        inverse_matrix = np.linalg.inv(rotation_matrix)
        transformed_point = np.dot(inverse_matrix, homogeneous_point)
        
        return transformed_point[:3]