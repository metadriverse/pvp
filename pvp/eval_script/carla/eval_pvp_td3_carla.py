#eval script for single carla ckpt change line 114 for pvp, old, TD3 model,
import argparse
import os
import os.path as osp
from collections import defaultdict

import pandas as pd
import json
import numpy as np
from pvp.sb3.common.monitor import Monitor
from pvp.eval_script.carla.carla_eval_utils import setup_model, setup_model_old
from pvp.experiments.carla.carla_env import HumanInTheLoopCARLAEnv


def load_human_data(path, data_usage=5000):
    """
   This method reads the states and actions recorded by human expert in the form of episode
   """
    with open(path, "r") as f:
        episode_data = json.load(f)["data"]
    np.random.shuffle(episode_data)
    assert data_usage < len(episode_data), "Data is not enough"
    data = {"state": [], "action": [], "next_state": [], "reward": [], "terminal": []}
    for cnt, step_data in enumerate(episode_data):
        if cnt >= data_usage:
            break
        data["state"].append(step_data["obs"])
        data["next_state"].append(step_data["new_obs"])
        data["action"].append(step_data["actions"])
        data["terminal"].append(step_data["dones"])
    # get images as features and actions as targets
    return data


def eval_one_checkpoint_random(model, eval_env, log_dir, num_episodes):
    model.set_parameters(model_path)
    count = 0
    recorder = defaultdict(list)
    try:
        obs = eval_env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            if done:
                count += 1
                for k, v in info.items():
                    recorder[k].append(v)
                print("The environment is terminated. Final info: ", info)
                if count >= num_episodes:
                    break
                obs = eval_env.reset()
    finally:
        eval_env.close()

    df = pd.DataFrame(dict(recorder))
    df.to_csv(osp.join(log_dir, "eval_result.csv"))

    print("=====\nFinish evaluating agent with checkpoint: ", model_path)
    print("=====\n")


def eval_one_checkpoint(model_path, model, eval_env, log_dir, num_episodes):
    model.set_parameters(model_path)
    count = 0
    recorder = defaultdict(list)
    try:
        obs = eval_env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            if done:
                count += 1
                for k, v in info.items():
                    recorder[k].append(v)
                print("The environment is terminated. Final info: ", info)
                if count >= num_episodes:
                    break
                obs = eval_env.reset()
    finally:
        eval_env.close()

    df = pd.DataFrame(dict(recorder))
    df.to_csv(osp.join(log_dir, "eval_result.csv"))

    print("=====\nFinish evaluating agent with checkpoint: ", model_path)
    print("=====\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="./eval", type=str, help="CKPT name.")
    parser.add_argument("--port", default=9000, type=int, help="Carla server port.")
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    args = parser.parse_args()
    port = args.port
    seed = args.seed
    ckpt = args.ckpt

    num_episodes = 15
    obs_mode = "birdview"

    # ===== Setup the training environment =====
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
    eval_env = Monitor(env=train_env, filename=None)
    model = setup_model(eval_env=eval_env, seed=seed, obs_mode=obs_mode)
    #setuo_model_old for old old, setup_modeltd3 for td3 baselines
    model_root_path = ckpt
    checkpoints = [p for p in os.listdir(model_root_path) if p.startswith("rl_model")]
    checkpoint_indices = sorted([int(p.split("_")[2]) for p in checkpoints], reverse=True)
    # eval_one_checkpoint(
    #     model=model, eval_env=eval_env, log_dir="../", num_episodes=num_episodes
    # )
    # for model_index in checkpoint_indices[::2]:
    for model_index in [24800]:
        model_path = os.path.join(model_root_path, "rl_model_{}_steps.zip".format(model_index))
        log_dir = model_path.replace(".zip", "")
        os.makedirs(log_dir, exist_ok=True)
        eval_one_checkpoint(
            model_path=model_path, model=model, eval_env=eval_env, log_dir=log_dir, num_episodes=num_episodes
        )
