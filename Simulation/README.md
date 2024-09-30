# Simulation

## Collecting Data

Run the following command to collect heuristic data control.

```bash
cd Simulation/crew-algorithms/crew_algorithms/
export PYTHONPATH=..:$PYTHONPATH
WANDB_MODE=disabled python collect_data envs.num_seekers=[num_seekers] envs.num_hiders=[num_hiders] envs.start_seed=[starting_seed] envs.num_games=[num_games]
```
<div style="background-color:#f0f0f0; padding:10px; border-radius:5px;">

In this game, the configuration options control key aspects of the simulation. Below is a description of each configuration option:

- **`num_seekers`**: Specifies the number of seekers in the game. Seekers are agents tasked with finding the hiders.

- **`num_hiders`**: Specifies the number of hiders in the game. Hiders are agents that try to evade the seekers.

- **`starting_seed`**: Sets the starting seed for the random number generator. This ensures that the game's randomness can be replicated for testing or debugging purposes.

- **`num_games`**: The total number of games or rounds to be played during data collection. Increasing this number allows for more extensive data to be gathered.

</div>
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

To evaluate the trained models and log the performance, run:

```bash
cd Simulation/crew-algorithms/crew_algorithms/
export PYTHONPATH=..:$PYTHONPATH
WANDB_MODE=disabled python test envs.num_seekers=[num_seekers] envs.num_hiders=[num_hiders] envs.start_seed=[starting_seed] envs.num_games=[num_games]
```
