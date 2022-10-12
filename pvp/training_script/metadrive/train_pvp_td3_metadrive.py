import argparse
import os
import os.path as osp

from pvp_iclr_release.utils.human_in_the_loop_env import HumanInTheLoopEnv
from pvp_iclr_release.stable_baseline3.common.callbacks import CallbackList, CheckpointCallback
from pvp_iclr_release.stable_baseline3.common.monitor import Monitor
from pvp_iclr_release.stable_baseline3.common.wandb_callback import WandbCallback
from pvp_iclr_release.pvp.pvp_td3.pvp_td3 import pvpTD3
from pvp_iclr_release.stable_baseline3.td3.policies import TD3Policy
from pvp_iclr_release.stable_baseline3.old.old_buffer import oldReplayBuffer
from pvp_iclr_release.utils.older_utils import get_time_str

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", default="TEST", type=str, help="The experiment name.")
    parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    parser.add_argument("--start-seed", default=100, type=int, help="The environment map random seed.")
    parser.add_argument("--control", default="joystick",choices=['joystick', 'xboxController', 'keyboard'] type=str, help="The experiment name.")
    
    args = parser.parse_args()

    # ===== Setup some meta information =====
    exp_name = args.exp_name
    seed = int(args.seed)
    start_seed = int(args.start_seed)
    use_wandb = args.wandb
    if not use_wandb:
        print("[WARNING] Please note that you are not using wandb right now!!!")

    project_name = "old_2022"
    team_name = "drivingforce"

    trial_name = "{}_seed{}_{}".format(exp_name, start_seed, get_time_str())
    log_dir = osp.join("runs", exp_name, trial_name)
    os.makedirs(osp.join("runs", exp_name), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    print("We start logging training data into {}".format(log_dir))

    # ===== Setup the config =====
    config = dict(
        # Environment config
        env_config={
            "manual_control": True,
            "use_render": True,
            "controller": args.control, # or joystick
            "window_size": (1600, 1100),
            "start_seed": start_seed,
            # "map": "COT",
            # "environment_num": 1,
        },

        # Algorithm config
        algo=dict(
            use_balance_sample=True,
            policy=TD3Policy,
            replay_buffer_class=oldReplayBuffer,  ###
            replay_buffer_kwargs=dict(
                discard_reward=True,  # xxx: We run in reward-free manner!
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

            learning_starts=100,  ###
            batch_size=100,  # Reduce the batch size for real-time copilot
            tau=0.005,
            gamma=0.99,
            # train_freq=1,
            # target_policy_noise=0,
            # policy_delay=1,

            action_noise=None,
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
        trial_name=trial_name,
        log_dir=log_dir
    )

    # ===== Setup the training environment =====
    train_env = HumanInTheLoopEnv(config=config["env_config"], )
    train_env = Monitor(env=train_env, filename=log_dir)
    config["algo"]["env"] = train_env
    assert config["algo"]["env"] is not None

    # ===== Setup the callbacks =====
    callbacks = [
        CheckpointCallback(
            name_prefix="rl_model",
            verbose=1,
            save_freq=100,
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
    model = pvpTD3(**config["algo"])

    # ===== Launch training =====
    model.learn(
        # training
        total_timesteps=4_0000,
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

        # save buffer
        buffer_save_timesteps = 5000,
        save_path_human = "./",
        save_path_replay = "./",
        save_buffer = True,

        # load buffer
        load_buffer = False,
        # load_path_human = "./human_buffer_100.pkl",
        # load_path_replay = "./replay_buffer_100.pkl",

        # cql warmup
        # warmup = True,
        # warmup_steps = 50000,
    )
