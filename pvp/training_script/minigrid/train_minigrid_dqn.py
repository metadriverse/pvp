import argparse
import os
import os.path as osp

import gym
import torch
from gym_minigrid.wrappers import ImgObsWrapper

from pvp_iclr_release.stable_baseline3.common.callbacks import CallbackList, CheckpointCallback
from pvp_iclr_release.stable_baseline3.common.monitor import Monitor
from pvp_iclr_release.stable_baseline3.common.vec_env import DummyVecEnv, VecFrameStack
from pvp_iclr_release.stable_baseline3.common.wandb_callback import WandbCallback
from pvp_iclr_release.stable_baseline3.dqn.dqn import DQN
from pvp_iclr_release.stable_baseline3.dqn.policies import CnnPolicy
from pvp_iclr_release.utils.older_utils import get_time_str
from pvp_iclr_release.training_script.minigrid.minigrid_env import MinigridWrapper, GrayScaleWrapper
from pvp_iclr_release.training_script.minigrid.minigrid_model import MinigridCNN, MinigridNet, FullObsMinigridPolicyNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", default="minigrid_dqn", type=str, help="The experiment name.")
    parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    parser.add_argument("--env-name", default="MiniGrid-Empty-Random-6x6-v0", type=str, help="Name of Gym environment")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--render", action="store_true", help="Whether to pop up window for visualization.")
    args = parser.parse_args()

    # ===== Setup some meta information =====
    exp_name = args.exp_name
    seed = int(args.seed)
    use_wandb = args.wandb
    if not use_wandb:
        print("[WARNING] Please note that you are not using wandb right now!!!")
    # obs_mode = args.obs_mode
    env_name = args.env_name
    render = args.render
    # control_mode = args.control
    lr = args.lr

    project_name = "old_2022"
    team_name = "drivingforce"

    trial_name = "{}_{}_lr{}_seed{}_{}".format(exp_name, env_name, lr, seed, get_time_str())
    log_dir = osp.join("../../old_minigrid/runs", exp_name, trial_name)
    os.makedirs(osp.join("../../old_minigrid/runs", exp_name), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    print("We start logging training data into {}".format(log_dir))

    # ===== Setup the config =====
    config = dict(

        # Environment config
        env_config=dict(),

        # Algorithm config
        algo=dict(
            policy=CnnPolicy,
            policy_kwargs=dict(
                features_extractor_class=MinigridCNN,
                activation_fn=torch.nn.Tanh,
                net_arch=[64, ]  # Remove FC in Q network
            ),

            env=None,
            optimize_memory_usage=True,

            # Hyper-parameters are collected from https://arxiv.org/pdf/1910.02078.pdf
            # MiniGrid specified parameters
            buffer_size=10_000,
            learning_rate=lr,
            # get_linear_fn(
            #     start=1e-4,
            #     end=1e-7,
            #     end_fraction=1.0,
            # ),

            exploration_fraction=0.30,  # Reach minimal exploration rate at 30% Total Steps
            exploration_final_eps=0.05,
            # gradient_steps=1,

            # === old set of hypers ===
            # learning_starts=1000,  # xxx: Original DQN has 100K warmup steps
            # batch_size=32,  # Reduce the batch size for real-time copilot
            # train_freq=4,
            # tau=1.0,
            # target_update_interval=1000,

            # === new set of hypers ===
            learning_starts=50,  # xxx: Original DQN has 100K warmup steps
            batch_size=256,  # Reduce the batch size for real-time copilot
            train_freq=1,
            tau=0.005,
            target_update_interval=1,

            gradient_steps=32,

            tensorboard_log=log_dir,
            create_eval_env=False,
            verbose=2,
            seed=seed,
            device="auto",
        ),

        # Meta data
        project_name=project_name,
        team_name=team_name,
        exp_name=exp_name,
        seed=seed,
        use_wandb=use_wandb,
        # obs_mode=obs_mode,
        trial_name=trial_name,
        log_dir=log_dir
    )

    # ===== Setup the training environment =====
    env = gym.make(env_name)
    env = Monitor(env=env, filename=log_dir)
    env = MinigridWrapper(env, enable_render=render, enable_human=False)
    env = ImgObsWrapper(env)
    train_env = VecFrameStack(DummyVecEnv([lambda: env]), n_stack=4)

    config["algo"]["env"] = train_env
    assert config["algo"]["env"] is not None

    # ===== Setup the callbacks =====
    callbacks = [
        CheckpointCallback(
            name_prefix="rl_model",
            verbose=1,
            save_freq=10000,
            save_path=osp.join(log_dir, "models")
        )
    ]
    if use_wandb:
        callbacks.append(
            WandbCallback(
                trial_name=trial_name,
                exp_name=exp_name,
                project_name=project_name,
                config=config
            )
        )
    callbacks = CallbackList(callbacks)

    # ===== Setup the training algorithm =====
    model = DQN(**config["algo"])

    # ===== Launch training =====
    model.learn(
        # training
        total_timesteps=100_000,
        callback=callbacks,
        reset_num_timesteps=True,

        # eval
        eval_env=None,
        eval_freq=-1,
        n_eval_episodes=2,
        eval_log_path=None,

        # logging
        tb_log_name=exp_name,  # Should place the algorithm name here!
        log_interval=1,
    )
