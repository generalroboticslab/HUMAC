# Simulation

## Collecting Data

Run the following command to collect heuristic data control.

```bash
cd Simulation/crew-algorithms/crew_algorithms/
export PYTHONPATH=..:$PYTHONPATH
WANDB_MODE=disabled python collect_data envs.num_seekers=[num_seekers] envs.num_hiders=[num_hiders] envs.start_seed=[starting_seed] envs.num_games=[num_games]
```
num_seekers is the number of seekers in the game, num_hiders is the number of hiders in the game, starting_seed is the starting seed, and num_games is the number of game to collect data.

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
