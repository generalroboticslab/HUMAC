import os
from time import time 

def test(num_seekers = 1, num_hiders = 1, start_seed = 1, num_games = 150, decision_frequency = 0.2, num_seekers_with_policy=0, model_path="", policy=""):
    # os.system("conda activate crew")
    os.system(f"WANDB_MODE=disabled python IL_Heuristic envs.num_seekers={num_seekers} envs.num_hiders={num_hiders} envs.start_seed={start_seed} envs.num_games={num_games} envs.decision_frequency={decision_frequency} envs.num_seekers_with_policy={num_seekers_with_policy} envs.model_path={model_path} envs.policy={policy}")

def collect_data_from_heuristic(num_seekers = 3, num_hiders = 2, start_seed = 451, num_games = 3, decision_frequency = 1):
    # os.system("conda activate crew")
    os.system(f"WANDB_MODE=disabled python collect_data envs.num_seekers={num_seekers} envs.num_hiders={num_hiders} envs.start_seed={start_seed} envs.num_games={num_games} envs.decision_frequency={decision_frequency}")

if __name__ == "__main__":

    decision_frequency = 0.2
    seeker_num_1 = [3,3,4,4]
    hider_num_ = [2,3,3,4]
    start_seed_list = [1,151,301]
    # num_games_l = [150,150,150]
    
    policy = "PE_H"
    model_path = "../../model_weights/FT_PE_H_short.pth"
    
    for seeker_num,hider_num in zip(seeker_num_1,hider_num_):
        for num_seekers_with_policy in range(2,3):
            assert num_seekers_with_policy <= seeker_num, f"There are only {i} Seekers, you cannot let {num_seekers_with_policy} seekers to run with trained policy"
            for i,start_seed in enumerate(start_seed_list):
                if i != len(start_seed_list) - 1:
                    num_games = start_seed_list[i+1] - start_seed
                else:
                    num_games = 451 - start_seed
                
                print(num_games)
                
                test(num_seekers=seeker_num,
                    num_hiders=hider_num,
                    start_seed=start_seed,
                    num_games=num_games,
                    decision_frequency=decision_frequency,
                    num_seekers_with_policy=num_seekers_with_policy,
                    model_path=model_path,
                    policy=policy,
                    )
            

            