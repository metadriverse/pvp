import argparse
import os
from pathlib import Path

from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
from pvp.pvp_td3 import HACOTD3
from pvp.sb3.td3.policies import TD3Policy
from pvp.sb3.common.callbacks import CallbackList, CheckpointCallback
from pvp.sb3.common.monitor import Monitor
from pvp.sb3.common.wandb_callback import WandbCallback
from pvp.sb3.haco import HACOReplayBuffer
from pvp.utils.utils import get_time_str

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", default="pvp_metadrive", type=str, help="The name for this batch of experiments.")
    parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    # parser.add_argument("--start-seed", default=100, type=int, help="The environment map random seed.")
    parser.add_argument(
        "--device",
        default="keyboard",
        choices=['wheel', 'gamepad', 'keyboard'],
        type=str,
        help="The control device, selected from [wheel, gamepad, keyboard]."
    )

    # TODO: Add a flag to control whether should record data.

    args = parser.parse_args()

    # ===== Set up some arguments =====
    control_device = args.device
    experiment_batch_name = args.exp_name
    seed = args.seed  # TODO: Should we set the random seed for env too?
    trial_name = "{}_{}_{}".format(experiment_batch_name, control_device, get_time_str())
    use_wandb = args.wandb
    if not use_wandb:
        print("[WARNING] Please note that you are not using wandb right now!!!")

    # TODO: What is this for?
    # project_name = "haco_2022"
    # team_name = "drivingforce"

    experiment_dir = Path("runs") / experiment_batch_name
    trial_dir = experiment_dir / trial_name
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(trial_dir, exist_ok=True)
    print(f"We start logging training data into {trial_dir}")

    # ===== Setup the config =====
    if control_device == "wheel":
        control_device_internal_name = "steering_wheel"
    elif control_device == "keyboard":
        control_device_internal_name = "keyboard"
    elif control_device == "gamepad":
        control_device_internal_name = "xbox"
    else:
        raise ValueError("Unknown control device {}".format(control_device))
    config = dict(
        # Environment config
        env_config={
            "manual_control": True,
            "use_render": True,
            "controller": control_device_internal_name,
            "window_size": (1600, 1100),
            "start_seed": 100,  # TODO: Check this config, add doc.
        },

        # Algorithm config
        algo=dict(
            use_balance_sample=True,
            policy=TD3Policy,
            replay_buffer_class=HACOReplayBuffer,
            replay_buffer_kwargs=dict(
                discard_reward=True,  # We run in reward-free manner!
                discard_takeover_start=False,
                takeover_stop_td=False
            ),
            policy_kwargs=dict(net_arch=[256, 256]),
            intervention_start_stop_td=True,
            env=None,
            learning_rate=1e-4,
            q_value_bound=1,
            optimize_memory_usage=True,
            buffer_size=50_000,  # We only conduct experiment less than 50K steps
            learning_starts=100,  # The number of steps before
            batch_size=100,  # Reduce the batch size for real-time copilot
            tau=0.005,
            gamma=0.99,
            # train_freq=1,
            # target_policy_noise=0,
            # policy_delay=1,
            action_noise=None,
            tensorboard_log=trial_dir,
            create_eval_env=False,
            verbose=2,
            seed=seed,
            device="auto",
        ),

        # Meta data
        # TODO: What is this?
        # project_name=project_name,
        # team_name=team_name,
        exp_name=experiment_batch_name,
        seed=seed,
        use_wandb=use_wandb,
        trial_name=trial_name,
        log_dir=str(trial_dir)
    )

    # ===== Setup the training environment =====
    train_env = HumanInTheLoopEnv(config=config["env_config"], )
    train_env = Monitor(env=train_env, filename=str(trial_dir))
    config["algo"]["env"] = train_env
    assert config["algo"]["env"] is not None

    # ===== Setup the callbacks =====
    callbacks = [
        CheckpointCallback(name_prefix="rl_model", verbose=1, save_freq=200, save_path=str(trial_dir / "models"))
    ]
    if use_wandb:
        # TODO: Test wandb later
        callbacks.append(
            WandbCallback(
                trial_name=trial_name, exp_name=experiment_batch_name, project_name=project_name, config=config
            )
        )
    callbacks = CallbackList(callbacks)

    # ===== Setup the training algorithm =====
    model = HACOTD3(**config["algo"])

    # ===== Launch training =====
    model.learn(
        # training
        total_timesteps=50_000,
        callback=callbacks,
        reset_num_timesteps=True,

        # eval
        eval_env=None,
        eval_freq=-1,
        n_eval_episodes=2,
        eval_log_path=None,

        # logging
        tb_log_name=experiment_batch_name,  # Should place the algorithm name here!
        log_interval=1,

        # TODO: How to process save buffer and load buffer?
        # save buffer
        buffer_save_timesteps=5000,
        save_path_human="./",
        save_path_replay="./",
        save_buffer=True,

        # load buffer
        load_buffer=False,
        # load_path_human = "./human_buffer_100.pkl",
        # load_path_replay = "./replay_buffer_100.pkl",

        # TODO: What is CQL warmup?
        # cql warmup
        # warmup = True,
        # warmup_steps = 50000,
    )
