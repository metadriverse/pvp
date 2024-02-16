"""
Training script for training PPO in MetaDrive Safety Env.
"""
import argparse
import os
from pathlib import Path

from pvp.sb3.common.callbacks import CallbackList, CheckpointCallback
from pvp.sb3.common.env_util import make_vec_env
from pvp.sb3.common.vec_env import DummyVecEnv, SubprocVecEnv
from pvp.sb3.common.wandb_callback import WandbCallback
from pvp.sb3.ppo import PPO
from pvp.sb3.ppo.policies import ActorCriticPolicy
from pvp.utils.utils import get_time_str


def register_env(make_env_fn, env_name):
    from gym.envs.registration import register
    register(id=env_name, entry_point=make_env_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="ppo_metadrive", type=str, help="The name for this batch of experiments.")
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    parser.add_argument("--ckpt", default=None, type=str, help="Path to previous checkpoint.")
    parser.add_argument("--debug", action="store_true", help="Set to True when debugging.")
    parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
    parser.add_argument("--wandb_project", type=str, default="", help="The project name for wandb.")
    parser.add_argument("--wandb_team", type=str, default="", help="The team name for wandb.")
    args = parser.parse_args()

    # FIXME: Remove this in future.
    if args.wandb_team is None:
        args.wandb_team = "drivingforce"
    if args.wandb_project is None:
        args.wandb_project = "pvp2024"

    # ===== Set up some arguments =====
    # control_device = args.device
    experiment_batch_name = "{}".format(args.exp_name)
    seed = args.seed
    trial_name = "{}_{}".format(experiment_batch_name, get_time_str())

    use_wandb = args.wandb
    project_name = args.wandb_project
    team_name = args.wandb_team
    if not use_wandb:
        print("[WARNING] Please note that you are not using wandb right now!!!")

    experiment_dir = Path("runs") / experiment_batch_name
    trial_dir = experiment_dir / trial_name
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(trial_dir, exist_ok=True)
    print(f"We start logging training data into {trial_dir}")

    # ===== Setup the config =====
    config = dict(
        # ===== Environment =====
        env_config=dict(
            use_render=False,  # Open the interface
            manual_control=False,  # Allow receiving control signal from external device
            # controller=control_device,
            # window_size=(1600, 1100),
            horizon=1500,
        ),
        num_train_envs=32,

        # ===== Environment =====
        eval_env_config=dict(
            use_render=False,  # Open the interface
            manual_control=False,  # Allow receiving control signal from external device
            start_seed=1000,
            horizon=1500,
        ),
        num_eval_envs=1,

        # ===== Training =====
        algo=dict(
            policy=ActorCriticPolicy,
            n_steps=1024,  # n_steps * n_envs = total_batch_size
            n_epochs=20,
            learning_rate=5e-5,
            batch_size=256,
            clip_range=0.1,
            vf_coef=0.5,
            ent_coef=0.0,
            max_grad_norm=10.0,
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
    vec_env_cls = SubprocVecEnv
    if args.debug:
        config["num_train_envs"] = 1
        config["algo"]["n_steps"] = 64
        vec_env_cls = DummyVecEnv

    # ===== Setup the training environment =====
    train_env_config = config["env_config"]


    def _make_train_env():
        from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
        from pvp.sb3.common.monitor import Monitor
        train_env = HumanInTheLoopEnv(config=train_env_config)
        train_env = Monitor(env=train_env, filename=str(trial_dir))
        return train_env


    train_env_name = "metadrive_train-v0"
    register_env(_make_train_env, train_env_name)
    train_env = make_vec_env(_make_train_env, n_envs=config["num_train_envs"], vec_env_cls=vec_env_cls)
    config["algo"]["env"] = train_env
    assert config["algo"]["env"] is not None

    # ===== Also build the eval env =====
    eval_env_config = config["eval_env_config"]


    def _make_eval_env():
        from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
        from pvp.sb3.common.monitor import Monitor
        eval_env = HumanInTheLoopEnv(config=eval_env_config)
        eval_env = Monitor(env=eval_env, filename=str(trial_dir))
        return eval_env


    eval_env_name = "metadrive_eval-v0"
    register_env(_make_eval_env, eval_env_name)
    eval_env = make_vec_env(_make_eval_env, n_envs=config["num_eval_envs"], vec_env_cls=vec_env_cls)

    # ===== Setup the callbacks =====
    save_freq = 100_000  # Number of steps per model checkpoint
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
    model = PPO(**config["algo"])

    if args.ckpt:
        ckpt = Path(args.ckpt)
        print(f"Loading checkpoint from {ckpt}!")
        from pvp.sb3.common.save_util import load_from_zip_file
        data, params, pytorch_variables = load_from_zip_file(
            ckpt, device=model.device, print_system_info=False
        )
        model.set_parameters(params, exact_match=True, device=model.device)

    # ===== Launch training =====
    model.learn(
        # training
        total_timesteps=10_000_000,
        callback=callbacks,
        reset_num_timesteps=True,

        # eval
        eval_env=eval_env,
        eval_freq=100_000,
        n_eval_episodes=100,
        eval_log_path=str(trial_dir),

        # logging
        tb_log_name=experiment_batch_name,
        log_interval=1,
        # save_buffer=False,
        # load_buffer=False,
    )
