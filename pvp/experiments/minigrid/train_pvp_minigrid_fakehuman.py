"""
Training script for training PVP in MiniGrid environment.
"""
import argparse
import os
from pathlib import Path

# import gymnasium as gym
import torch

from pvp.experiments.minigrid.minigrid_env import MiniGridMultiRoomN2S4, MiniGridMultiRoomN4S5, \
    MiniGridEmpty6x6, wrap_minigrid_env, MiniGridEmpty16x16
from pvp.experiments.minigrid.minigrid_model import MinigridCNN
from pvp.pvp_dqn import PVPDQN
from pvp.pvp_dqn_cpl import PVPDQNCPL
from pvp.sb3.common.callbacks import CallbackList, CheckpointCallback
from pvp.sb3.common.monitor import Monitor
from pvp.sb3.common.wandb_callback import WandbCallback
from pvp.sb3.dqn.policies import CnnPolicy
from pvp.utils.shared_control_monitor import SharedControlMonitor
from pvp.utils.utils import get_time_str

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name", default="pvp_minigrid_fakehuman", type=str, help="The name for this batch of experiments."
    )
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
    parser.add_argument("--use_fake_human_with_failure", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="", help="The project name for wandb.")
    parser.add_argument("--wandb_team", type=str, default="", help="The team name for wandb.")

    parser.add_argument(
        "--env",
        default="emptyroom",
        type=str,
        help="Nick name of the environment.",
        choices=["emptyroom", "emptyroom16", "tworoom", "fourroom"]
    )
    args = parser.parse_args()

    # ===== Set up some arguments =====
    use_fake_human_with_failure = args.use_fake_human_with_failure
    env_name = args.env
    experiment_batch_name = "{}_{}".format(args.exp_name, env_name)
    seed = args.seed
    trial_name = "{}_{}".format(experiment_batch_name, get_time_str())

    use_wandb = args.wandb
    project_name = args.wandb_project
    team_name = args.wandb_team
    if not use_wandb:
        print("[WARNING] Please note that you are not using wandb right now!!!")

    experiment_dir = Path("runs") / experiment_batch_name
    trial_dir = experiment_dir / trial_name
    eval_log_dir = trial_dir / "evaluations"
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(trial_dir, exist_ok=True)
    print(f"We start logging training data into {trial_dir}")

    # ===== Setup the config =====
    config = dict(

        # Environment config
        env_config=dict(),

        # Algorithm config
        algo=dict(
            policy=CnnPolicy,
            policy_kwargs=dict(features_extractor_class=MinigridCNN, activation_fn=torch.nn.Tanh, net_arch=[
                64,
            ]),

            # === PVP setting ===
            replay_buffer_kwargs=dict(discard_reward=True),  # PZH: We run in reward-free manner!
            exploration_fraction=0.0,  # 1% * 100k = 1k
            exploration_initial_eps=0.0,
            exploration_final_eps=0.0,
            env=None,
            optimize_memory_usage=True,

            # Hyper-parameters are collected from https://arxiv.org/pdf/1910.02078.pdf
            # MiniGrid specified parameters
            buffer_size=10_000,
            learning_rate=1e-4,

            # === New hypers ===
            learning_starts=10,  # PZH: Original DQN has 100K warmup steps
            batch_size=32,
            train_freq=(1, 'step'),
            tau=0.005,
            target_update_interval=1,
            gradient_steps=32,
            tensorboard_log=trial_dir,
            create_eval_env=False,
            verbose=2,
            seed=seed,
            device="auto",
        ),

        # Experiment log
        exp_name=experiment_batch_name,
        seed=seed,
        use_wandb=use_wandb,
        trial_name=trial_name,
        log_dir=str(trial_dir)
    )

    # ===== Setup the training environment =====
    if env_name == "emptyroom":
        env_class = MiniGridEmpty6x6
    elif env_name == "emptyroom16":
        env_class = MiniGridEmpty16x16
    elif env_name == "tworoom":
        env_class = MiniGridMultiRoomN2S4
    elif env_name == "fourroom":
        env_class = MiniGridMultiRoomN4S5
    else:
        raise ValueError("Unknown environment: {}".format(env_name))

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # env = wrap_minigrid_env(env_class, enable_takeover=True)
    env = wrap_minigrid_env(
        env_class, enable_takeover=False, use_fake_human=True, use_fake_human_with_failure=use_fake_human_with_failure
    )
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    env = Monitor(env=env, filename=str(trial_dir))

    # TODO: Remove this for fakehuman experiment.
    # env = SharedControlMonitor(env=env, folder=trial_dir / "data", prefix=trial_name, save_freq=100)

    train_env = env

    # ===== Also build the eval env =====
    def _make_eval_env():
        env = wrap_minigrid_env(env_class, enable_takeover=False)
        env = Monitor(env=env, filename=str(trial_dir))
        return env

    eval_env = _make_eval_env()
    config["algo"]["env"] = train_env
    assert config["algo"]["env"] is not None

    # ===== Setup the callbacks =====
    save_freq = 500  # Number of steps per model checkpoint
    callbacks = [
        CheckpointCallback(name_prefix="rl_model", verbose=1, save_freq=save_freq, save_path=str(trial_dir / "models"))
    ]
    if use_wandb:
        callbacks.append(
            WandbCallback(
                trial_name=trial_name,
                exp_name=experiment_batch_name,
                team_name=team_name,
                project_name=project_name,
                config=config
            )
        )
    callbacks = CallbackList(callbacks)

    # ===== Setup the training algorithm =====
    model = PVPDQN(**config["algo"])

    # ===== Launch training =====
    model.learn(
        # training
        total_timesteps=50_000,
        callback=callbacks,
        reset_num_timesteps=True,

        # eval
        eval_env=eval_env,
        eval_freq=10,  # Evaluate every 20 steps in training.
        n_eval_episodes=20,
        eval_log_path=str(trial_dir),

        # logging
        tb_log_name=experiment_batch_name,
        log_interval=1,
    )
