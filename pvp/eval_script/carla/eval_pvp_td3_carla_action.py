# load ckpt for old and pvp, store their action given certain obs(obs from straight road)
import argparse
import os
import os.path as osp
from collections import defaultdict
import gym
from gym import spaces
import torch
import pandas as pd
import json
import numpy as np
from pvp.sb3.common.monitor import Monitor
from pvp.eval_script.carla.carla_eval_utils import setup_model, setup_model_old
from pvp.experiments.carla.carla_env import HumanInTheLoopCARLAEnv


# class DummyEnv(gym.Env):
#     def __init__(self):
#         self.action_space = spaces.Box(np.array[-1,-1], np.array[1,1], dtype=np.float32)
#         self.observation_space= spaces.Dict({"image": spaces.Box(low=0, high=255, shape=(84, 84, 3))})
def load_human_data(path, data_usage=5000):
    """
   This method reads the states and actions recorded by human expert in the form of episode
   """
    with open(path, "r") as f:
        episode_data = json.load(f)["data"]
    np.random.shuffle(episode_data)
    # assert data_usage < len(episode_data), "Data is not enough"
    data = {"state": [], "action": [], "next_state": [], "reward": [], "terminal": []}
    for cnt, step_data in enumerate(episode_data):
        if cnt >= data_usage:
            break
        data["state"].append(step_data["obs"])
        # data["speed"]
        data["next_state"].append(step_data["new_obs"])
        data["action"].append(step_data["actions"])
        data["terminal"].append(step_data["dones"])
    # get images as features and actions as targets
    return data

    df = pd.DataFrame(dict(recorder))
    df.to_csv(osp.join(log_dir, "eval_result.csv"))

    print("=====\nFinish evaluating agent with checkpoint: ", model_path)
    print("=====\n")


if __name__ == '__main__':
    port = 9000
    seed = 0
    datapath = "/home/xxx/model/human_traj_1.json"
    obs_mode = "birdview"
    train_env = HumanInTheLoopCARLAEnv(
        config=dict(
            obs_mode=obs_mode,
            force_fps=0,
            disable_vis=False,  # xxx: @xxx, change this to disable/open vis!
            debug_vis=False,
            port=port,
            disable_takeover=True,
            controller="keyboard",
            env={"visualize": {
                "location": "lower right"
            }}
        )
    )
    ckpt_index_list = []
    mean1 = []
    std1 = []
    mean2 = []
    std2 = []
    mean3 = []
    std3 = []
    mean4 = []
    std4 = []
    for ckpt_index in range(1000, 22000, 1000):
        print("dealing with ckpt index: " + str(ckpt_index))
        model_path1 = "/home/xxx/model/compare/models/rl_model_" + str(ckpt_index) + "_steps"
        model_path2 = "/home/xxx/model/oldold_carla/CARLA-OLDold_birdview_seed0_2022-06-15_17-17-29/models/rl_model_" + str(
            ckpt_index
        ) + "_steps.zip"

        model1 = setup_model(eval_env=train_env, seed=seed, obs_mode=obs_mode)
        model1.set_parameters(model_path1)
        model2 = setup_model_old(eval_env=train_env, seed=seed, obs_mode=obs_mode)
        model2.set_parameters(model_path2)
        count = 0
        recorder = defaultdict(list)
        data = load_human_data(datapath)
        obs = data["state"]
        obs = np.array(obs)
        # obs = torch.from_numpy(obs)
        true_action = data["action"]
        true_action = np.array(true_action)
        # true_action = torch.from_numpy(true_action)
        steer_list1 = []
        steer_list2 = []
        throttle_list1 = []
        throttle_list2 = []
        for i in range(len(obs)):
            currobs = obs[i]
            curr_true_act = true_action[i]
            dictobs = dict({"image": currobs, "speed": [0.5]})
            curr_agent_action1, _states = model1.predict(dictobs, deterministic=True)
            curr_agent_action2, _states = model2.predict(dictobs, deterministic=True)
            # print(curr_agent_action)
            # print("yeah")
            steer_list1.append(curr_agent_action1[0])
            steer_list2.append(curr_agent_action2[0])
            throttle_list1.append(curr_agent_action1[1])
            throttle_list2.append(curr_agent_action2[1])
        ckpt_index_list.append(ckpt_index)
        std1.append(np.std(steer_list1))
        mean1.append(np.mean(steer_list1))
        std2.append(np.std(steer_list2))
        mean2.append(np.mean(steer_list2))
        mean3.append(np.mean(throttle_list1))
        std3.append(np.std(throttle_list1))
        mean4.append(np.mean(throttle_list2))
        std4.append(np.std(throttle_list2))
    results = {
        "ckpt_index": ckpt_index_list,
        "pvp_steer_mean": mean1,
        "pvp_steer_std": std1,
        "pvp_throttle_mean": mean3,
        "pvp_throttle_std": std3,
        "old_steer_mean": mean2,
        "old_steer_std": std2,
        "old_throttle_mean": mean4,
        "old_throttle_std": std4
    }
    df = pd.DataFrame(results)
    df.to_csv("/home/xxx/model/finalcompare.csv")
    import matplotlib.pyplot as plt
    plt.plot(ckpt_index_list, mean3, label="pvp")
    plt.plot(ckpt_index_list, mean4, label="old")
    plt.legend()
    plt.show()
    plt.show()
