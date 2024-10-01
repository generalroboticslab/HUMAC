import hydra
from attrs import define
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torchrl.trainers.helpers.collectors import OnPolicyCollectorConfig
from crew_algorithms.envs.configs import EnvironmentConfig, register_env_configs
from crew_algorithms.utils.wandb_utils import WandbConfig
import os
import time


green = "\033[92m"
red = "\033[91m"
yellow = "\033[93m"
blue = "\033[94m"
reset = "\033[0m"

@define(auto_attribs=True)
class Config:
    envs: EnvironmentConfig = MISSING
    """Settings for the environment to use."""
    collector: OnPolicyCollectorConfig = OnPolicyCollectorConfig(frames_per_batch=1)
    """Settings to use for the on-policy collector."""
    wandb: WandbConfig = WandbConfig(project="random")
    """WandB logger configuration."""
    collect_data: bool = True
    """Whether or not to collect data and save a new dataset to WandB."""

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
register_env_configs()

@hydra.main(version_base=None, config_path="../conf", config_name="Build")
def get_info(cfg: Config) -> None:
    global num_seekers
    num_seekers = cfg['envs']['num_seekers']
    global num_hiders
    num_hiders = cfg['envs']['num_hiders']
    global start_seed
    start_seed = cfg['envs']['start_seed']
    global num_games 
    num_games = cfg['envs']['num_games']
    global data_folder
    data_folder = cfg['envs']['data_folder']


@hydra.main(version_base=None, config_path="../conf", config_name="Build")
def collect_data(cfg: Config) -> None:
    import os
    import uuid
    from pathlib import Path
    import torch
    import wandb
    from torchrl.record.loggers import generate_exp_name, get_logger
    from crew_algorithms.envs.channels import ToggleTimestepChannel
    from crew_algorithms.collect_data.utils import (
        make_env,
        make_agent,
        save_images,
    )
    from crew_algorithms.utils.rl_utils import make_collector
   
    device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    toggle_timestep_channel = ToggleTimestepChannel(uuid.uuid4())
    
    device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    toggle_timestep_channel = ToggleTimestepChannel(uuid.uuid4())
    env_fn = lambda: make_env(cfg.envs, toggle_timestep_channel, device)
    policy = make_agent('cuda',num_seekers)
    


    step = 0
    
    seed_list = range(start_seed,start_seed+num_games+1)
    total_game = 0
    total_data = 0
    
    while total_game < num_games:
    #make the collector
        random_seed = seed_list[total_game]
        base_directory = f"../../{data_folder}/{num_seekers}Seeker_vs_{num_hiders}Hider/Seed={random_seed}/"
        
        collector = make_collector(cfg.collector, env_fn, policy, device)
        
        for batch, data in enumerate(collector):
            next_id = data['next','agents','observation','obs_3'][0][0][2].item()
            hider_left = data['next','agents','observation','obs_3'][0][0][-1].item()
            save_images(
                cfg,
                data,
                base_directory,
                cfg.collector.frames_per_batch,
                batch,
            )

            if hider_left == 0 or next_id > 1 or batch == (120/cfg['envs']['decision_frequency']-1):
                total_game += 1
                collector.shutdown()
                total_data += batch*num_seekers
                print(f"{red}{num_seekers} Seekers{reset} VS {green}{num_hiders} Hiders{reset}, Num of test: {yellow}{total_game}/{num_games}{reset}, Total Data Collected: {total_data}")
                break
                
    print("Collect Data Finished")

if __name__ == "__main__":
    get_info()

    #write all the testing seed to the txt file
    file_path =  str(os.getcwd())+"/random_seed.txt"
    with open(file_path, 'w') as file:
        pass
    with open(file_path, 'a') as file:
        for i in range(start_seed, num_games+start_seed):
            file.write(str(i) + '\n')
            file.write(str(i) + '\n')
    
    collect_data()
