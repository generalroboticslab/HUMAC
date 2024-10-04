# Real-World Experiment

## Hardware setup 

[Vicon Motion Tracking System](https://www.vicon.com/) <br>

[Robot](https://www.dji.com/robomaster-ep)

[RGB Camera](https://www.amazon.com/WyreStorm-Presenter-Noise-canceling-Conference-Education/dp/B09TJZSFJ9/ref=sr_1_2_sspa?crid=29KGZMDETE448&dib=eyJ2IjoiMSJ9.K7Mz1N5pED0yn1EBz_7tsMmJcTWqCLzANK6lnVj6yZ9V5uD4kc1B7-4I_xnWFgH8gJldoz3WBoSzdvsUCGDbSlYtoeLfJnz06w3noIexm9s5SuyaB7MUWGp6FxGD7kwjQcmmq5LeJ4beEwjbdsg5AbRet795shZuA0v7c70FSqrn-nsn7yiEJHx4SEQBo1tbUHznFT8kRHZRhLzboiNZQmwzdy47Davce5h4dUZOhhY.77xNXPE3SPziEOYvcN4Hf0WESYgW5xne4JkEi39Z9Nk&dib_tag=se&keywords=webcam+4k+fisheye&qid=1727971531&sprefix=webcam+4k+fisheye%2Caps%2C81&sr=8-2-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1)

## Install the dependency 
```bash
cd Real-World
pip install -r requirements.txt
```
## Start Listening Data from Vicon System

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.sh
roslaunch vicon_bridge vicon.launch
```

## Name Robot and Prepare the Json File

The key is the ip address of the robot, and the value is a list with [Unique ID on robot smart controller, agent name]. Please save the json file as robot_info.json into the folder:

```plaintext
HUMAC/Real-World/environment/
```
The agent name must be like Seeker{ID} or Hider{ID}. The ID should start from 1 and increase as the number of robots increases. An example robot_info.json will be like:

```plaintext
{"192.168.0.123":["159CKCH0070ACX","Seeker1"],
"192.168.0.154":["159CL170070WES","Seeker2"],
"192.168.0.166":["159CL170070CDD","Seeker3"],
"192.168.0.188":["159CL170070CVT","Hider1"],
"192.168.0.199":["159CKCH0070WEF","Hider2"],
"192.168.0.100":["159CL170070GTE","Hider3"]}
```

## Connect Robot to Wifi
Power up the DJI robomaster, and change the smart controller on it to connect to router.
Then run the following command:

```bash
cd Real-World/environment/
python connect_robot_to_wifi.py 
```

After successfully connecting to wifi, there will be a prompt in the terminal, as well as a sound from the speaker of the robot.

## Collecting Data

To collect data, prepare the amount of robot you need. For example, if you want to collect collect 2 Seekers vs 1 Hider data, you need to power up Seeker1, Seeker2, and Hider1. The ID of the robot you power matters in this case. 

### Heuristic Data

Run the following command to start the process:

```bash
cd Real-World/environment/
python collect_heuristic_data.py  
```

The prompt will be like this 

1. **Setup the arena**:

![Step1](../images/step1.png)

Put only the obstacles in the arena then press '1' and enter.


2. **Spawn the Robots**:

![Step2](../images/step2.png)

Put all the robots in the arena then press '2' to start the episode.

3. **Take out the caught hider**:

![Step3](../images/step3.png)

As the game proceed, if any hider robot is caught, the whole game will be paused and you need to take the caught one (The Led Light of the hider robot will turn from green to blue) out of the arena. After that, press '3' and enter to resume the game.



### Human Data
For collecting human data, run 

```bash
cd Real-World/environment/
python collect_human_data.py  
```

Everything else stay the same. Use the UI popped up to guide the robots. Left-click to set robot destination and right-click to switch between robots.

## Trained Policies
If you wish to use the policies we trained and skip the training and fine-tuning, you can download them [here](https://drive.google.com/drive/folders/1PD8xUyoZI92qHqNpSBPHLH5rsxumaeYC?usp=sharing). 


## Training and Fine-tuning

To train the policies, please run the corresponding script in the training folder and feel free to tune the parameters.

## Evaluation

To evaluate the trained models run the following:

```bash
cd Real-World/environment/
python test_policy.py
```

