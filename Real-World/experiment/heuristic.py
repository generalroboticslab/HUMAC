import rospy
from geometry_msgs.msg import TransformStamped
from functools import partial
import math
import numpy as np

wall_normal_vector = {1:(-1,0),2:(0,-1),3:(1,0),4:(0,1)}
wall_direction_vector = {1:(0,-1),2:(1,0),3:(0,1),4:(-1,0)}

def find_box_corners(x_c, y_c, d_x, d_y, length_x, length_y):
    # Normalize the direction vector
    norm = np.sqrt(d_x**2 + d_y**2)
    d_x, d_y = d_x / norm, d_y / norm
    
    # Perpendicular vector
    p_x, p_y = -d_y, d_x
    
    # Half-lengths
    v_x, v_y = (length_x / 2) * d_x, (length_x / 2) * d_y
    w_x, w_y = (length_y / 2) * p_x, (length_y / 2) * p_y
    
    # Corners of the rectangle
    top_right = (x_c + v_x + w_x, y_c + v_y + w_y)
    top_left = (x_c - v_x + w_x, y_c - v_y + w_y)
    bottom_left = (x_c - v_x - w_x, y_c - v_y - w_y)
    bottom_right = (x_c + v_x - w_x, y_c + v_y - w_y)
    
    return top_right, top_left, bottom_left, bottom_right

def is_point_in_triangle(p, a, b, c):
        # Vector cross product to determine the point's position relative to triangle edges
        def sign(o, p1, p2):
            return (o[0] - p2[0]) * (p1[1] - p2[1]) - (p1[0] - p2[0]) * (o[1] - p2[1])
        
        d1 = sign(p, a, b)
        d2 = sign(p, b, c)
        d3 = sign(p, c, a)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

def is_point_in_box(point_x, point_y, corners):
    
    # Break the rectangle into two triangles and check if the point is in either
    top_right, top_left, bottom_left, bottom_right = corners
    is_in_triangle1 = is_point_in_triangle((point_x, point_y), top_right, top_left, bottom_left)
    is_in_triangle2 = is_point_in_triangle((point_x, point_y), top_right, bottom_right, bottom_left)
    
    return is_in_triangle1 or is_in_triangle2

def quaternion_to_dictional_vector(quaternion):
    """
    Convert a quaternion into Euler angles (roll, pitch, yaw)
    Quaternion is assumed to be in the format (w, x, y, z)
    """
    x, y, z, w = quaternion
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    direction_x = math.cos(yaw)
    direction_y = math.sin(yaw)
    
    return direction_x, direction_y

def gamestate_listener(path,seeker_info_dict, hider_info_dict):
    # Use functools.partial to pass the extra argument
    rospy.Subscriber(path, TransformStamped, partial(callback, agent_name=path.split("/")[-1] , seeker_info_dict=seeker_info_dict, hider_info_dict=hider_info_dict))

def callback(data, agent_name,seeker_info_dict, hider_info_dict):
    # Now you can use both 'data' and 'extra_arg' in your callback
    location = (data.transform.translation.x, data.transform.translation.y)
    direction = quaternion_to_dictional_vector((data.transform.rotation.x, data.transform.rotation.y, data.transform.rotation.z, data.transform.rotation.w))
    agent_info_dict = seeker_info_dict if "Seeker" in agent_name else hider_info_dict
    agent_info_dict[agent_name] = {"location": location, "direction": direction}

def avoid_wall(location,direction,action,map_info, obstacle_detection_range = 1,rotation_angle = 30):
    direction_nomralized = normalize(direction)
    left_direction = direction_nomralized
    right_direction = direction_nomralized
    

    for i in range(180//rotation_angle):
        #turn left loop
        ray_final_position = (location[0] + left_direction[0]*obstacle_detection_range, location[1] + left_direction[1]*obstacle_detection_range)
        ray_final_position_half = (location[0] + left_direction[0]*obstacle_detection_range/2, location[1] + left_direction[1]*obstacle_detection_range/2)
        direction_new = left_direction
        
        #rotate the direction left and right for 30 degrees and shoot both rays 
        ray_left_direction = rotate_angle(left_direction,rotation_angle,False)
        ray_right_direction = rotate_angle(left_direction,rotation_angle,True)
        
        ray_left_final_position = (location[0] + ray_left_direction[0]*obstacle_detection_range, location[1] + ray_left_direction[1]*obstacle_detection_range)
        ray_right_final_position = (location[0] + ray_right_direction[0]*obstacle_detection_range, location[1] + ray_right_direction[1]*obstacle_detection_range)
        
        ray_left_final_position_half = (location[0] + ray_left_direction[0]*obstacle_detection_range/2, location[1] + ray_left_direction[1]*obstacle_detection_range/2)
        ray_right_final_position_half = (location[0] + ray_right_direction[0]*obstacle_detection_range/2, location[1] + ray_right_direction[1]*obstacle_detection_range/2)
        
        if check_wall(ray_final_position_half) and check_wall(ray_left_final_position_half) and check_wall(ray_right_final_position_half) and check_obstacle(ray_final_position,map_info) and check_obstacle(ray_left_final_position,map_info) and check_obstacle(ray_right_final_position,map_info) and check_obstacle(ray_final_position_half,map_info) and check_obstacle(ray_left_final_position_half,map_info) and check_obstacle(ray_right_final_position_half,map_info):
            if i == 0:
                pass
            else:
                action = ray_final_position
            break
        
        left_direction = ray_left_direction
        
        #turn right loop
        
        ray_final_position = (location[0] + right_direction[0]*obstacle_detection_range, location[1] + right_direction[1]*obstacle_detection_range)
        ray_final_position_half = (location[0] + right_direction[0]*obstacle_detection_range/2, location[1] + right_direction[1]*obstacle_detection_range/2)
        direction_new = right_direction
        #rotate the direction right and right for 30 degrees and shoot both rays 
        ray_left_direction = rotate_angle(right_direction,rotation_angle,False)
        ray_right_direction = rotate_angle(right_direction,rotation_angle,True)
        
        ray_left_final_position = (location[0] + ray_left_direction[0]*obstacle_detection_range, location[1] + ray_left_direction[1]*obstacle_detection_range)
        ray_right_final_position = (location[0] + ray_right_direction[0]*obstacle_detection_range, location[1] + ray_right_direction[1]*obstacle_detection_range)
        
        ray_left_final_position_half = (location[0] + ray_left_direction[0]*obstacle_detection_range/2, location[1] + ray_left_direction[1]*obstacle_detection_range/2)
        ray_right_final_position_half = (location[0] + ray_right_direction[0]*obstacle_detection_range/2, location[1] + ray_right_direction[1]*obstacle_detection_range/2)
        
        if check_wall(ray_final_position_half) and check_wall(ray_left_final_position_half) and check_wall(ray_right_final_position_half) and check_obstacle(ray_final_position,map_info) and check_obstacle(ray_left_final_position,map_info) and check_obstacle(ray_right_final_position,map_info) and check_obstacle(ray_final_position_half,map_info) and check_obstacle(ray_left_final_position_half,map_info) and check_obstacle(ray_right_final_position_half,map_info):
            if i == 0:
                pass
            else:
                action = ray_final_position
            break
        
        right_direction = ray_right_direction
    
    return action,direction_new

def count_number_of_wall(location,obstacle_detection_range = 0.5):
    
    num_of_wall = 0
    if location[0] < -2.5 + obstacle_detection_range or location[0] > 2.5 - obstacle_detection_range:
        num_of_wall += 1
    if location[1] < -2.5 + obstacle_detection_range or location[1] > 2.5 - obstacle_detection_range: 
        num_of_wall += 1
    
    return num_of_wall

def get_wall_number(location,obstacle_detection_range = 0.5):
    wall_number_list = []
    if location[0] < -2.5 + obstacle_detection_range:
        wall_number_list.append(1)
    if location[0] > 2.5 - obstacle_detection_range:
        wall_number_list.append(3)
    if location[1] < -2.5 + obstacle_detection_range:
        wall_number_list.append(2)
    if location[1] > 2.5 - obstacle_detection_range:
        wall_number_list.append(4)
    
    return wall_number_list

def check_all_seeker_on_same_side(location,seeker_location_list,num_of_wall):
    """
    Check if all the seeker are on the same side of the hider
    """
    vector_list = []
    for seeker_location in seeker_location_list:
        vector = (seeker_location[0] - location[0], seeker_location[1] - location[1])
        vector_list.append(vector)
    
    positivecrossproduct = 0
    negativecrossproduct = 0
    
    all_seeker_same_side = False
    
    if num_of_wall == 0:
        
        for reference_vector in vector_list:
            for vector in vector_list:
                crossproduct = reference_vector[0]*vector[1] - reference_vector[1]*vector[0]
                if crossproduct > 0:
                    positivecrossproduct += 1
                elif crossproduct < 0:
                    negativecrossproduct += 1
        
            all_seeker_same_side = positivecrossproduct == 0 or negativecrossproduct == 0
            if all_seeker_same_side:
                break
            
            positivecrossproduct = 0
            negativecrossproduct = 0
        
        seeker1_index = 0
        seeker2_index = 0
        
        max_angle = float("-inf")
        
        if all_seeker_same_side:
            for i in range(len(vector_list)):
                for j in range(i+1,len(vector_list)):
                    angle = angle_between_vectors(vector_list[i],vector_list[j])
                    if angle > max_angle:
                        max_angle = angle
                        seeker1_index = i
                        seeker2_index = j
        
        else:
            
            for i in range(len(vector_list)):
                for j in range(i+1,len(vector_list)):
                    angle = angle_between_vectors(vector_list[i],vector_list[j])
                    if angle > max_angle and check_adjacent(vector_list,i,j,angle) and both_seeker_not_next_to_wall(vector_list[i],vector_list[j]):
                        max_angle = angle
                        seeker1_index = i
                        seeker2_index = j               
            
    #one wall
    elif num_of_wall == 1:
        wall_number = get_wall_number(location)[0]
        reference_vector = wall_normal_vector[wall_number]
        
        for vector in vector_list:
            crossproduct = reference_vector[0]*vector[1] - reference_vector[1]*vector[0]
            if crossproduct > 0:
                positivecrossproduct += 1
            elif crossproduct < 0:
                negativecrossproduct += 1
            
        all_seeker_same_side = positivecrossproduct == 0 or negativecrossproduct == 0

        max_gap = float("-inf")
        seeker1_index = 0
        seeker2_index = 0
        
        if not all_seeker_same_side:
            for i in range(len(vector_list)):
                for j in range(i+1,len(vector_list)):
                    gap = get_distance(vector_list[i],vector_list[j])
                    angle = angle_between_vectors(vector_list[i],vector_list[j])
                    if gap > max_gap and check_adjacent(vector_list,i,j,angle) and check_wall_in_middle(vector_list,i,j,wall_number):
                        max_gap = gap
                        seeker1_index = i
                        seeker2_index = j
                
                wall_gap = 2.5 - abs(max(seeker_location_list[i][0],seeker_location_list[i][1]))    
                
                if wall_gap > max_gap and check_adjacent_to_wall(seeker_location_list,vector_list,i,wall_number):
                    seeker1_index = i
                    seeker2_index = -1

    #corner
    else:
        wall_number_list = get_wall_number(location)
        wall1_norm = wall_normal_vector[wall_number_list[0]]
        wall2_norm = wall_normal_vector[wall_number_list[1]]
        
        reference_vector = (-wall1_norm[0] - wall2_norm[0], -wall1_norm[1] - wall2_norm[1])
        
        for vector in vector_list:
            crossproduct = reference_vector[0]*vector[1] - reference_vector[1]*vector[0]
            if crossproduct > 0:
                positivecrossproduct += 1
            elif crossproduct < 0:
                negativecrossproduct += 1
        
        if positivecrossproduct == 0 or negativecrossproduct == 0:
            all_seeker_same_side = True
        
        max_gap = float("-inf")
        seeker1_index = 0
        seeker2_index = 0
        
        for i in range(len(vector_list)):
            for j in range(i+1,len(vector_list)):
                gap = get_distance(vector_list[i],vector_list[j])
                angle = angle_between_vectors(vector_list[i],vector_list[j])
                
                if gap > max_gap and check_adjacent(vector_list,i,j,angle) and check_wall_in_middle(vector_list,i,j,wall_number_list[0]) and check_wall_in_middle(vector_list,i,j,wall_number_list[1]):
                    max_gap = gap
                    seeker1_index = i
                    seeker2_index = j
            
            wall1_number = wall_number_list[0]
            wall2_number = wall_number_list[1]
            
            wall1_gap = 2.5 - abs(dot_product(wall_normal_vector[wall1_number],vector_list[i]))
            wall2_gap = 2.5 - abs(dot_product(wall_normal_vector[wall2_number],vector_list[i]))
            
            if wall1_gap < wall2_gap:
                wall_number = wall1_number
                wall_index = -1
            else:
                wall_number = wall2_number
                wall_index = -2
            
            if wall1_gap > max_gap and check_adjacent_to_wall(seeker_location_list,vector_list,i,wall_number):
                seeker1_index = i
                seeker2_index = wall_index
                
                
    return all_seeker_same_side,seeker1_index,seeker2_index

def dot_product(vector_a, vector_b):
    return sum(a * b for a, b in zip(vector_a, vector_b))

def magnitude(vector):
    return math.sqrt(sum(a ** 2 for a in vector))

def check_wall(position):
    
    if position[0] < -2.5 or position[0] > 2.5 or position[1] < -2.5 or position[1] > 2.5:
        return False
    else:
        return True
        
def check_obstacle(position,map_info):
    result = True
    for four_corner in map_info:
        if is_point_in_box(position[0],position[1],four_corner):
            result = False
            break
    return result
    
def rotate_angle(direction,angle,clockwise):
    x,y = direction
    angle_rad = math.radians(angle)
    
    # Determine the rotation direction
    if clockwise:
        angle_rad = -angle_rad

    # Compute the new coordinates after rotation
    new_x = x * math.cos(angle_rad) - y * math.sin(angle_rad)
    new_y = x * math.sin(angle_rad) + y * math.cos(angle_rad)

    return (new_x, new_y)

def angle_between_vectors(vector_a, vector_b):
    dot_prod = dot_product(vector_a, vector_b)
    magnitude_a = magnitude(vector_a)
    magnitude_b = magnitude(vector_b)
    cos_theta = dot_prod / (magnitude_a * magnitude_b)
    # Ensure the value is within the valid range for acos due to floating-point precision
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    angle_radians = math.acos(cos_theta)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

def get_distance(vector1,vector2):
    return math.sqrt((vector1[0] - vector2[0])**2 + (vector1[1] - vector2[1])**2)

def check_adjacent(vector_list,i,j,angle):
    result = True
    for k in range(len(vector_list)):
        if k != i and k != j:
            angle1 = angle_between_vectors(vector_list[i],vector_list[k])
            angle2 = angle_between_vectors(vector_list[j],vector_list[k])
            if abs(angle1+angle2 - angle) < 0.5:
                result = False
                break
    return result

def both_seeker_not_next_to_wall(vector1,vector2):
    if vector1[0] < -1.8 or vector1[0] > 1.8 or vector1[1] < -1.8 or vector1[1] > 1.8:
        return False
    if vector2[0] < -1.8 or vector2[0] > 1.8 or vector2[1] < -1.8 or vector2[1] > 1.8:
        return False
    return True

def calculate_run_away_angle(vector1,vector2,run_in_middle = False):
    """
    Calculate the angle to run away from 2 seekers
    """
    if run_in_middle:
        return (vector1[0] + vector2[0], vector1[1] + vector2[1])
    else:
        return (-vector1[0] - vector2[0], -vector1[1] - vector2[1])
    
def check_wall_in_middle(vector_list,i,j,wall_number):
    """
    Check if the wall is in the middle of the 2 seeker
    """
    wall_normal = wall_normal_vector[wall_number]
    angle1 = angle_between_vectors(vector_list[i],vector_list[j])
    angle2 = angle_between_vectors(vector_list[i],wall_normal)
    angle3 = angle_between_vectors(vector_list[j],wall_normal)
    
    if abs(angle1 - angle2 - angle3) < 1:
        return False
    
    return True

def check_adjacent_to_wall(seeker_location_list,vector_list,i,wall_number):
    wall_normal = wall_normal_vector[wall_number]
    for j in range(len(vector_list)):
        if j != i:
            if (wall_normal[0] * vector_list[j][0] + wall_normal[1] * vector_list[j][1]) * (wall_normal[0] * vector_list[i][0] + wall_normal[1] * vector_list[i][1]) >= 0:
                current_gap = 2.5 - abs(max(seeker_location_list[i][0],seeker_location_list[i][1]))
                compare_gap = 2.5 - abs(max(seeker_location_list[j][0],seeker_location_list[j][1]))
                if current_gap > compare_gap:
                    return False
                    break
    return True

def seeker_heuristic(seeker_info_dict, hider_info_dict,hider_alreay_die_list,seeker_detection_range = 3, obstacle_detection_range = 1,map_info=None):
    """
    Seeker heuristic function
    """
    
    action_dict = {} 

    for agent_name in seeker_info_dict.keys():
        
        action = None # initialize to go straight
        
        if seeker_info_dict[agent_name] == {}:
            continue
        
        location = seeker_info_dict[agent_name]["location"]
        direction = seeker_info_dict[agent_name]["direction"]
        
        #detect the hider in the range and pick the closest one as the target to chase
        min_distance = float("inf")
        target = None
        for hider_name, hider_info in hider_info_dict.items():
            if hider_name in hider_alreay_die_list:
                continue
            if hider_info == {}:
                continue
            hider_location = hider_info["location"]
            if hider_location[0] > location[0] - seeker_detection_range/2 and hider_location[0] < location[0] + seeker_detection_range/2 and hider_location[1] > location[1] - seeker_detection_range/2 and hider_location[1] < location[1] + seeker_detection_range/2:
                distance = math.sqrt((hider_location[0] - location[0])**2 + (hider_location[1] - location[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    target = hider_name
        
        #recalculate the action
        if target is not None:
            action = hider_info_dict[target]["location"]
            direction = (action[0] - location[0], action[1] - location[1])
        action,_ = avoid_wall(location,direction,action,obstacle_detection_range = obstacle_detection_range, rotation_angle=30,map_info=map_info)

        action_dict[agent_name] = action
    # print(f"Seeker heuristic: {action_dict}")
        
    return action_dict
               
def hider_heuristic(hider_info_dict, seeker_info_dict, seeker_detection_range = 2.5, obstacle_detection_range = 1,map_info=None):
    
    """
    Hider heuristic function
    """
    
    action_dict = {}
    for agent_id,hider_name in enumerate(hider_info_dict.keys()):
        action = None #Need to change
        
        target_list = []
    
        if hider_info_dict[hider_name] == {}:
            continue
        
        location = hider_info_dict[hider_name]["location"]
        direction = hider_info_dict[hider_name]["direction"]
        
        #Detect if any seeker is in the range
        for seeker_name, seeker_info in seeker_info_dict.items():
            if seeker_info == {}:
                continue
            seeker_location = seeker_info["location"]
            distance = math.sqrt((seeker_location[0] - location[0])**2 + (seeker_location[1] - location[1])**2)
            
            if distance < seeker_detection_range/2:
                target_list.append(seeker_name)
        
        #Calculate the next action
        if target_list != []:
            
            #only one seeker
            if len(target_list) == 1:
                # print("1 Seeker is found")
                target = target_list[0]
                direction = (location[0] - seeker_info_dict[target]["location"][0], location[1] - seeker_info_dict[target]["location"][1])
            #multiple seeker
            else:
                
                # print("Multiple Seeker in Wall")
                #if not in wall 
                seeker_location_list = [seeker_info_dict[seeker_name]["location"] for seeker_name in target_list]
                if count_number_of_wall(location) == 0:
                    
                    all_on_same_side,seeker1_index,seeker2_index= check_all_seeker_on_same_side(location,seeker_location_list,num_of_wall=0)
                    #print(f"Multiple Seeker open space, All on the same side of the wall: {all_on_same_side}")
                    seeker1_location = seeker_location_list[seeker1_index]
                    seeker2_location = seeker_location_list[seeker2_index]
                    
                    vector1 = (seeker1_location[0] - location[0], seeker1_location[1] - location[1])
                    vector2 = (seeker2_location[0] - location[0], seeker2_location[1] - location[1])
                    
                    if all_on_same_side:
                        direction = calculate_run_away_angle(vector1,vector2,run_in_middle=False)
                    
                    else:
                        direction = calculate_run_away_angle(vector1,vector2,run_in_middle=True)
    
                #if in wall
                else:
                    #multiple seeker in wall
                    if count_number_of_wall(location) == 1:
                        all_on_same_side,seeker1_index,seeker2_index= check_all_seeker_on_same_side(location,seeker_location_list,num_of_wall=1)
    
                        if all_on_same_side:
                            seeker1_location = seeker_location_list[0]
                            seeker2_location = seeker_location_list[1]
                            
                            vector1 = (seeker1_location[0] - location[0], seeker1_location[1] - location[1])
                            vector2 = (seeker2_location[0] - location[0], seeker2_location[1] - location[1])
                            
                            direction = calculate_run_away_angle(vector1,vector2,run_in_middle=False)
                        #not all seeker on one side of the wall
                        else:
                            run_in_middle = True
                            #run in the middle of seeker and wall
                            if seeker2_index == -1:
                                
                                seeker_location = seeker_location_list[seeker1_index]
                                
                                wall_number = get_wall_number(location)[0]
                                wall_d = wall_direction_vector[wall_number]
                                
                                seeker_vector = (seeker_location[0] - location[0], seeker_location[1] - location[1])
                                
                                if wall_d[0] * seeker_vector[0] + wall_d[1] * seeker_vector[1] >= 0:
                                    direction = wall_d
                                else:
                                    direction = (-wall_d[0],-wall_d[1])
                                
                                
                            
                            #run away from 2 seekers
                            else:
                                seeker1_location = seeker_location_list[seeker1_index]
                                seeker2_location = seeker_location_list[seeker2_index]
                                
                                vector1 = (seeker1_location[0] - location[0], seeker1_location[1] - location[1])
                                vector2 = (seeker2_location[0] - location[0], seeker2_location[1] - location[1])

                                direction = calculate_run_away_angle(vector1,vector2,run_in_middle=True)
                                #print(f"Calculated: {direction}")
                    
                    #multiple seeker in corner
                    else:
                        all_on_same_side,seeker1_index,seeker2_index= check_all_seeker_on_same_side(location,seeker_location_list,num_of_wall=2)
                        
                        
                        if all_on_same_side:
                            seeker1_location = seeker_location_list[0]
                            seeker2_location = seeker_location_list[1]
                            
                            vector1 = (seeker1_location[0] - location[0], seeker1_location[1] - location[1])
                            vector2 = (seeker2_location[0] - location[0], seeker2_location[1] - location[1])
                            
                            direction = calculate_run_away_angle(vector1,vector2,run_in_middle=False)
                        else:
                            run_in_middle = True
                            #print(f"Seeker1 Index: {seeker1_index}, Seeker2 Index: {seeker2_index}")
                            if seeker2_index > 0:
                                seeker1_location = seeker_location_list[seeker1_index]
                                seeker2_location = seeker_location_list[seeker2_index]
                                
                                vector1 = (seeker1_location[0] - location[0], seeker1_location[1] - location[1])
                                vector2 = (seeker2_location[0] - location[0], seeker2_location[1] - location[1])
                                
                                direction = calculate_run_away_angle(vector1,vector2,run_in_middle=True)
                            elif seeker2_index == -1:
                                seeker_location = seeker_location_list[seeker1_index]
                                
                                wall_number = get_wall_number(location)[0]
                                wall_d = wall_direction_vector[wall_number]
                                
                                seeker_vector = (seeker_location[0] - location[0], seeker_location[1] - location[1])
                                
                                if wall_d[0] * seeker_vector[0] + wall_d[1] * seeker_vector[1] >= 0:
                                    direction = wall_d
                                else:
                                    direction = (-wall_d[0],-wall_d[1])
                            else:
                                seeker_location = seeker_location_list[seeker1_index]
                                
                                wall_number = get_wall_number(location)[1]
                                wall_d = wall_direction_vector[wall_number]
                                
                                seeker_vector = (seeker_location[0] - location[0], seeker_location[1] - location[1])
                                
                                if wall_d[0] * seeker_vector[0] + wall_d[1] * seeker_vector[1] >= 0:
                                    direction = wall_d
                                else:
                                    direction = (-wall_d[0],-wall_d[1])
                        # print(f"Direction: {direction}\n")
                        
            
            direction = normalize(direction)
            action = (location[0] + 2*direction[0], location[1] + 2*direction[1])
            
        #Avoid the walls if there are any 
        action,direction_new = avoid_wall(location,direction,action,obstacle_detection_range = obstacle_detection_range, rotation_angle=30,map_info=map_info) 
        direction_new = normalize(direction_new)
        
        #Run into the seeker while into the wall
        if count_number_of_wall(location,obstacle_detection_range = obstacle_detection_range) == 1:
            #running away from seeker direction and avoid wall direction are opposite 
            if sum(a*b for a,b in zip(direction_new,direction)) < 0:
                action = (location[0] + direction[0], location[1] + direction[1])

        action_dict[hider_name] = action
       
    return action_dict
    
def heuristic_policy(seeker_info_dict, hider_info_dict, hider_alreay_die_list, map_info):
    """
    Heuristic policy function
    """
    seeker_action_dict = seeker_heuristic(seeker_info_dict, 
                                          hider_info_dict, 
                                          hider_alreay_die_list,
                                          map_info = map_info)
    hider_action_dict = hider_heuristic(hider_info_dict, 
                                        seeker_info_dict,

                                        map_info = map_info)
    #combine 2 dictionaries
    action_dict = {**seeker_action_dict, **hider_action_dict}
    return action_dict

def normalize(vector):
    """
    Normalize a vector
    """
    magnitude = math.sqrt(vector[0]**2 + vector[1]**2)
    return (vector[0]/magnitude, vector[1]/magnitude)
