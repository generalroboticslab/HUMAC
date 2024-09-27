import hydra
from attrs import define
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torchrl.trainers.helpers.collectors import OnPolicyCollectorConfig
from crew_algorithms.envs.configs import EnvironmentConfig, register_env_configs
from crew_algorithms.utils.wandb_utils import WandbConfig
import os


green = "\033[92m"
red = "\033[91m"
yellow = "\033[93m"
blue = "\033[94m"
reset = "\033[0m"

global num_of_frames
num_of_frames = 5 

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
def get_info(cfg: Config):
    global base_directory
    base_directory = f"../../{cfg['envs']['num_seekers']}Seeker_vs_{cfg['envs']['num_hiders']}Hider_ts_2"
    global num_seekers
    num_seekers = cfg['envs']['num_seekers']
    global num_hiders
    num_hiders = cfg['envs']['num_hiders']
    global start_seed
    start_seed = cfg['envs']['start_seed']
    global num_games 
    num_games = cfg['envs']['num_games']
    global decision_frequency
    decision_frequency = cfg['envs']['decision_frequency']
    global num_seekers_with_policy
    num_seekers_with_policy = cfg['envs']['num_seekers_with_policy']
    global base_policy
    base_policy = cfg['envs']['base_policy']
    global addon_policy
    addon_policy = cfg['envs']['addon_policy']
    global base_model_path
    base_model_path = cfg['envs']['base_model_path']
    global addon_model_path
    addon_model_path = cfg['envs']['addon_model_path']
    
    
@hydra.main(version_base=None, config_path="../conf", config_name="Build")
def test(cfg: Config):
    import os
    import uuid
    from pathlib import Path
    import torch
    import wandb
    import time
    from torchrl.record.loggers import generate_exp_name, get_logger
    from crew_algorithms.envs.channels import ToggleTimestepChannel
    from crew_algorithms.test.utils import (
        make_env,
        make_agent,
    )
    from crew_algorithms.utils.rl_utils import make_collector
    
    #get the random seed of the current run
   
    #set up the environment
    device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    toggle_timestep_channel = ToggleTimestepChannel(uuid.uuid4())
    env_fn = lambda: make_env(cfg.envs, toggle_timestep_channel, device)
    
    model = make_agent('cuda',num_of_frames,num_seekers,num_seekers_with_policy,decision_frequency,base_policy,addon_policy)
    
    #load base model weights 
    print("\n")
    if base_policy != "Heuristic":
        if base_model_path != "" and os.path.exists(base_model_path):
            weights = torch.load(base_model_path)['model_state_dict']    
            model.base_policy.load_state_dict(weights)
            print(f"{green}Base Policy Weights Loaded{reset}")
        else:
            print(f"{red}Base model is empty{reset}")
    else:
        print(f"{green}Base model is Heuristic{reset}")
    
    #load addon model weights
    if addon_model_path.find("_split_") == -1:
        if addon_model_path != "" and os.path.exists(addon_model_path):
            weights = torch.load(addon_model_path)['model_state_dict']    
            model.addon_policy.load_state_dict(weights)
            print(f"{green}Addon Policy Weights Loaded{reset}")
        else:
            print(f"{red}Addon model is empty{reset}")
    else:
        addon_model_path_list = addon_model_path.split("_split_")
        teammate_path = addon_model_path_list[0]
        self_path = addon_model_path_list[1]
        
        teammate_weight = torch.load(teammate_path)['model_state_dict']
        model.addon_policy.teammate_prediction_model.load_state_dict(teammate_weight )


        model.addon_policy.load_state_dict(torch.load(self_path)['model_state_dict'],strict=False)
        print(f"{green}Addon Policy Weights Loaded{reset}")


    print("\n")

    #set up the directory
    base_directory = f"../../{num_seekers}Seeker({num_seekers - num_seekers_with_policy}+{num_seekers_with_policy})_vs_{num_hiders}Hider_{base_model_path.split('/')[-1][:-4]}+{addon_model_path.split('/')[-1][:-4]}"
    
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)
    file_path = base_directory+"/game_result.txt"
    
    #run the test
    last_hider_num = num_hiders
    global time_list
    time_list = []
    step = 0
    
    total_time = 0
    start_time = time.time()
    total_game = 0
    total_success = 0
    
    seed_list = range(start_seed,start_seed+num_games+1)
    
    max_game_per_batch = 30
    num_games_left = num_games
    total_game_current_batch = 0
    
    while total_game_current_batch < min(max_game_per_batch,num_games_left):
    #make the collector
    
        collector = make_collector(cfg.collector, env_fn, model, device)
        
        for _, data in enumerate(collector):
            step += 1
            
            hider_left = data['next','agents','observation','obs_3'][0][0][-1].item()
            done = data['next','agents','done'][0][0].item()

            if last_hider_num - hider_left > 0:
                episode_time = round((step+1)*decision_frequency,1)
                time_list.append(episode_time)
            last_hider_num = hider_left
            
            if done:
                for _ in range (num_hiders - len(time_list)):
                    episode_time = round((step+1)*decision_frequency,1)
                    time_list.append(episode_time)
                
                with open(file_path, 'a') as file:
                    file.write(f"{seed_list[total_game]}: {str(time_list)}"+"\n")
                
                if time_list[-1] < 120:
                    total_success += 1
                
                total_time += time_list[-1]
                total_game_current_batch += 1
                total_game += 1
                
                print(f"{red}{num_seekers} Seekers{reset} VS {green}{num_hiders} Hiders{reset}, Num of test: {yellow}{total_game}/{num_games}{reset}, Success Rate: {green}{total_success/(total_game)*100:.2f}%{reset}, Average Episode Time: {blue}{total_time/(total_game):.2f}s{reset}, Total Running Time: {time.time()-start_time:.2f}s")
                        
                time_list = []
                step = 0
                
                if total_game_current_batch == min(max_game_per_batch,num_games_left):
                    collector.shutdown()
                    total_game_current_batch = 0
                    num_games_left -= min(max_game_per_batch, num_games_left)
                    break

    print("Test Finished")
        
if __name__ == "__main__":
    
    get_info()
    
    #write all the testing seed to the txt file
    file_path =  str(os.getcwd())+"/random_seed.txt"
    with open(file_path, 'w') as file:
        pass
    with open(file_path, 'a') as file:
        for i in range(start_seed, num_games+start_seed):
            file.write(str(i) + '\n')

    test()
        