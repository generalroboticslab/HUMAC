# Simulation

Our simulation can only Linux machine with [CREW](https://generalroboticslab.github.io/crew-docs/) correctly setup.

## Download Game Builds
Please download the game builds [here](https://drive.google.com/drive/folders/1Z8GJvNse7anBsv67tYMU9W0OUWAqNjOm?usp=drive_link) and put them into the folder:

```bash
HUMAC/Simulation/environment/
```

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


## Training and Fine-tuning

We have four policies in the paper, which are **`IL / IL-Long`**, **`PE-N`**, **`PE-H`**, and **`PE-T`**. You can train each of these policies by running the respective training script. 

If you wish to use the policies we trained, you can download them [here](https://drive.google.com/drive/folders/1_xfZlow_IGRBIz0-5BSLC76ATBmR-kfN?usp=drive_link). After you download them, please move them under the model_weights folder to make sure the evaluation runs smoothly.

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

All the models will be saved under the model_weights folder. 

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

- **`base_policy_name`**: The basic policy that the seekers are controlled by. The popular choices are "Heuristic", "IL", and "IL-Long". The configiration must be one of the follwing {Heuristic", "IL", "IL-Long","IL-FT","IL-Long-FT","PE-N","PE-H","PE-T"}

- **`addon_policy_name`**: The add-on policy that the seekers are controlled by. The configiration must be one of the follwing {Heuristic", "IL", "IL-Long","IL-FT","IL-Long-FT","PE-N","PE-H","PE-T"}

- **`num_seekers_with_addon_policy`**: The number of seekers with add on policy. The rest will be controlled by base policy. 

</div>

All the testing result will be logged to the folder test_results. To reproduce the result we provided in the paper, set **`starting_seed`** to be 1 and **`num_games`** to 450.

