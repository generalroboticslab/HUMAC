# Real-World Experiment

## Hardware setup 

[Vicon Motion Tracking System](https://www.vicon.com/) <br>

[Robot](https://www.dji.com/robomaster-ep)

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
HUMAC\Real-World\environment\
```
The agent name must be like Seeker{ID} or Hider{ID}. An example robot_info.json will be like:

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
cd Real-World
python3 environment/connect_robot_to_wifi.py 
```

After successfully connecting to wifi, there will be a prompt in the terminal, as well as a sound from the speaker of the robot.


## Collecting Data

### Heuristic Data

Run the following command to collect heuristic data control.

Change to the algorithm folder:
```bash
cd Simulation/crew-algorithms/crew_algorithms/
```

Export python path:
```bash
export PYTHONPATH=..:$PYTHONPATH
conda activate crew
```

Collect data:
```bash
WANDB_MODE=disabled python collect_data envs.num_seekers=[num_seekers] envs.num_hiders=[num_hiders] envs.start_seed=[starting_seed] envs.num_games=[num_games] envs.data_folder=[\path\to\save\data] envs.time_scale=[time_scale]
```

<div style="background-color:#f0f0f0; padding:10px; border-radius:5px;">

Below is a description of each configuration option:

- **`num_seekers`**: Specifies the number of seekers in the game. Seekers are agents tasked with finding the hiders.

- **`num_hiders`**: Specifies the number of hiders in the game. Hiders are agents that try to evade the seekers.

- **`starting_seed`**: Sets the starting seed for the random number generator. This ensures that the game's randomness can be replicated for testing or debugging purposes.

- **`num_games`**: The total number of games or rounds to be played during data collection. Increasing this number allows for more extensive data to be gathered.

- **`\path\to\save\data`**: Path to save the dataset.

- **`time_scale`**: The time scale of the simulation. For collecting heuristic data, the time scale is recommended to be <= 6. For collecting human guidance data, it is recommended to be <= 2.
</div>

### Human Data
For collecting human data, you need to open a client and join the game to guide the seekers, everything else stay the same.

## Trained Policies
If you wish to use the policies we trained and skip the training and fine-tuning, you can download them [here](https://drive.google.com/drive/folders/1_xfZlow_IGRBIz0-5BSLC76ATBmR-kfN?usp=drive_link). After you download them, please move them under the 
```bash
HUMAC\Simulation\model_weights
```
folder to make sure the evaluation runs smoothly.


## Training and Fine-tuning

We have four policies in the paper, which are **`IL / IL-Long`**, **`PE-N`**, **`PE-H`**, and **`PE-T`**. You can train each of these policies by running the respective training script. 

To train any of the policies, follow the steps below:

1. **Navigate to the training directory**:
    ```bash
    cd Simulation/training/
    ```
    
2. **Run the training script**:
    Use the appropriate policy name in place of **`{policy_name}`** (**`IL`**, **`PE-N`**, **`PE-H`**, or **`PE-T`**), and provide any specific arguments as needed:
   ```bash
    WANDB_MODE=disabled python train_{policy_name}.py --seed_value [seed_value] --batch_size [batch_size] --learning_rate [learning_rate] --epochs [epochs] --num_of_frames [num_of_frames] --step_ahead [step_ahead] --data_root_folder [path/to/data] 
   ```

3. **Fine-tuning**:
    ```bash
    --seed_value [seed_value] --batch_size [batch_size] --learning_rate [learning_rate] --epochs [epochs] --num_of_frames [num_of_frames] --step_ahead [step_ahead] --data_root_folder [path/to/data] --model [model_name]
    ```

4. **Available command-line arguments**:
    - **`--seed_value`**: The seed value for randomness to ensure reproducibility (default: 42).
    - **`--batch_size`**: The batch size for the data loader (default: 128).
    - **`--learning_rate`**: The learning rate for the optimizer (default: 0.001).
    - **`--epochs`**: The number of training epochs (default: 150).
    - **`--num_workers`**: The number of CPU workers used for data loading (default: 1).
    - **`--num_of_frames`**: The number of frames to stack (default: 5).
    - **`--step_ahead`**: The number of steps ahead for prediction. Setting step_ahead to be 1 is training IL, and making it > 1 is predicting longer than 1 step which refers to IL-Long in the paper.
    - **`--data_root_folder`**: The root folder path for the dataset (default: "path/to/IL/data").
    - **`--model`**: The name of the model to fine-tune.

All the models will be saved under the 
```bash
HUMAC\Simulation\model_weights
```
folder. 

### Notes:
- **Weights & Biases (WandB)**: If you're using [WandB](https://wandb.ai/), ensure you set **`WANDB_MODE=online`** and have your API key configured. If you want to disable it, you can use **`WANDB_MODE=disabled`** as shown in the examples.

## Evaluation

To evaluate the trained models run the following:

```bash
cd Simulation/crew-algorithms/crew_algorithms/
export PYTHONPATH=..:$PYTHONPATH
WANDB_MODE=disabled python test envs.num_seekers=[num_seekers] envs.num_hiders=[num_hiders] envs.start_seed=[starting_seed] envs.num_games=[num_games] envs.base_policy=[base_policy_name] envs.addon_policy=[addon_policy_name] envs.num_seekers_with_policy=[num_seekers_with_addon_policy]
```
<div style="background-color:#f0f0f0; padding:10px; border-radius:5px;">

Below is a description of each configuration option:

- **`num_seekers`**: Specifies the number of seekers in the game. Seekers are agents tasked with finding the hiders.

- **`num_hiders`**: Specifies the number of hiders in the game. Hiders are agents that try to evade the seekers.

- **`starting_seed`**: Sets the starting seed for the random number generator. This ensures that the game's randomness can be replicated for testing or debugging purposes.

- **`num_games`**: The total number of games or rounds to be played during data collection. Increasing this number allows for more extensive data to be gathered.

- **`base_policy_name`**: The basic policy that the seekers are controlled by. The popular choices are "Heuristic", "IL", and "IL-Long". The configiration must be one of the follwing {"Heuristic", "IL", "IL-Long","IL-FT","IL-Long-FT","PE-N","PE-H","PE-T"}

- **`addon_policy_name`**: The add-on policy that the seekers are controlled by. The configiration must be one of the follwing {"Heuristic", "IL", "IL-Long","IL-FT","IL-Long-FT","PE-N","PE-H","PE-T"}

- **`num_seekers_with_addon_policy`**: The number of seekers with add on policy. The rest will be controlled by base policy. 

</div>

All the testing result will be logged to the folder 
```bash
HUMAC/Simulation/test_results
```
folder.
 To reproduce the result we provided in the paper, set **`starting_seed`** to be 1 and **`num_games`** to 450. Our testing result can be found [here](https://drive.google.com/drive/folders/1fOJgRqlxBFC0VGEhBA50dXefr5-JooDF?usp=sharing)

