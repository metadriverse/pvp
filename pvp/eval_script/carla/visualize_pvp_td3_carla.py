# script for visualize carla step by step, for pvp
import argparse
import os
import os.path as osp
from collections import defaultdict

import pandas as pd

from pvp.sb3.common.monitor import Monitor
from pvp.eval_script.carla.carla_eval_utils import setup_model, setup_model_td3
from pvp.experiments.carla.carla_env import HumanInTheLoopCARLAEnv


def eval_one_checkpoint(model_path, model, eval_env, log_dir, num_episodes):
    model.set_parameters(model_path)
    count = 0
    recorder = defaultdict(list)
    try:
        obs = eval_env.reset()
        obs = eval_env.reset()
        obs = eval_env.reset()
        obs = eval_env.reset()
        obs = eval_env.reset()
        obs = eval_env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            # print(action)
            obs, reward, done, info = eval_env.step(action)
            a = 1
            if done:
                print("Model ckpt: " + str(model_path) + "Finish episode: " + str(count))
                for k, v in info.items():
                    recorder[k].append(v)
                print("The environment is terminated. Final info: ", info)
                break
                # obs = eval_env.reset()
    finally:
        eval_env.close()


if __name__ == '__main__':
    port = 9000
    seed = 0
    model_path = "/home/xxx/nvme/iclr_ckpt/carla_pvp/rl_model_23000_steps.zip"
    log_dir = "/home/xxx/nvme/iclr-visual/carla_pvp/"
    # model_path = "/home/xxx/nvme/iclr_ckpt/carla_td3/rl_model_1000000_steps.zip"
    # log_dir = "/home/xxx/nvme/iclr-visual/carla_td3/"
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
    eval_env.seed(0)
    model = setup_model(eval_env=eval_env, seed=seed, obs_mode=obs_mode)

    os.makedirs(log_dir, exist_ok=True)
    eval_one_checkpoint(
        model_path=model_path, model=model, eval_env=eval_env, log_dir=log_dir, num_episodes=num_episodes
    )
