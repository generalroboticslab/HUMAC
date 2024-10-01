# Simulation

Our simulation can only Linux machine with [CREW](https://generalroboticslab.github.io/crew-docs/) correctly setup.

## Download Game Builds
Please download the game builds [here]() and put them into the folder:

```bash
HUMAC/Simulation/environment/Crew version/crew-dojo/Builds
```

## Collecting Data

Run the following command to collect heuristic data control.

```bash
cd Simulation/crew-algorithms/crew_algorithms/
export PYTHONPATH=..:$PYTHONPATH
WANDB_MODE=disabled python collect_data envs.num_seekers=[num_seekers] envs.num_hiders=[num_hiders] envs.start_seed=[starting_seed] envs.num_games=[num_games]
```

<div style="background-color:#f0f0f0; padding:10px; border-radius:5px;">

Below is a description of each configuration option:

- **`num_seekers`**: Specifies the number of seekers in the game. Seekers are agents tasked with finding the hiders.

- **`num_hiders`**: Specifies the number of hiders in the game. Hiders are agents that try to evade the seekers.

- **`starting_seed`**: Sets the starting seed for the random number generator. This ensures that the game's randomness can be replicated for testing or debugging purposes.

- **`num_games`**: The total number of games or rounds to be played during data collection. Increasing this number allows for more extensive data to be gathered.

</div>

All the data will be collected to the folder.

## Training 
Run the following command to train the model.

```bash
cd Simulation/training/
python train.py
```
## Fine-tuning
Run the following command to fine-tune the model.

```bash
cd Simulation/training/
python Fine-tune.py
```

## Evaluation

To evaluate the trained models run:

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

