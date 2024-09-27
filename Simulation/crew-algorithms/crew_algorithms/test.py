import os
from time import time 

def test(num_seekers = 1, num_hiders = 1, start_seed = 1, num_games = 150, decision_frequency = 0.2, num_seekers_with_policy=0, base_policy="", addon_policy="", base_model_path="", addon_model_path=""):
    # os.system("conda activate crew")
    os.system(f"WANDB_MODE=disabled python test envs.num_seekers={num_seekers} envs.num_hiders={num_hiders} envs.start_seed={start_seed} envs.num_games={num_games} envs.decision_frequency={decision_frequency} envs.num_seekers_with_policy={num_seekers_with_policy} envs.base_policy={base_policy} envs.addon_policy={addon_policy} envs.base_model_path={base_model_path} envs.addon_model_path={addon_model_path}")

def collect_data_from_heuristic(num_seekers = 3, num_hiders = 2, start_seed = 451, num_games = 3, decision_frequency = 1):
    # os.system("conda activate crew")
    os.system(f"WANDB_MODE=disabled python collect_data envs.num_seekers={num_seekers} envs.num_hiders={num_hiders} envs.start_seed={start_seed} envs.num_games={num_games} envs.decision_frequency={decision_frequency}")

if __name__ == "__main__":

    
    decision_frequency = 0.2
    seeker_num_1 = [3,3,4]
    hider_num_l = [2,3,3]
    start_seed_1 = [[1,1,1],[1,241],[367]]
    end_seed = 451

    policy_number_list = [range(1,4),range(1,3),range(4,5)]
    
    base_policy = "Heuristic"
    addon_policy = "PE_N"
    base_policy_path= "../../model_weights/H.pth"
    addon_policy_path = "../../model_weights/Teammate_prediction.pth_split_../../model_weights/FT_PE_N.pth"

    
    for seeker_num,hider_num,start_seed1,p in zip(seeker_num_1,hider_num_l,start_seed_1,policy_number_list):
        for num_seekers_with_policy,start_seed in zip(p,start_seed1):
            assert num_seekers_with_policy <= seeker_num, f"There are only {seeker_num} Seekers, you cannot let {num_seekers_with_policy} seekers to run with trained policy"
            num_games= end_seed - start_seed 
            test(num_seekers=seeker_num,
                num_hiders=hider_num,
                start_seed=start_seed,
                num_games=num_games,
                decision_frequency=decision_frequency,
                num_seekers_with_policy=num_seekers_with_policy,
                base_policy=base_policy,
                addon_policy=addon_policy,
                base_model_path=base_policy_path,
                addon_model_path=addon_policy_path)
                
            

            