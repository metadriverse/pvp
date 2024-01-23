import argparse
import os
import os.path as osp

from drivingforce.haco_2022.train_metadrive.human_in_the_loop_env import HumanInTheLoopEnv
from pvp.sb3.common.callbacks import CallbackList, CheckpointCallback
from pvp.sb3.common.monitor import Monitor
from pvp.sb3.common.wandb_callback import WandbCallback
from pvp.sb3.haco import HACOPolicy, HACOReplayBuffer, HACO
from drivingforce.haco_2022.utils import get_time_str

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", default="MD-HACO", type=str, help="The experiment name.")
    parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    args = parser.parse_args()

    # ===== Setup some meta information =====
    exp_name = args.exp_name
    seed = int(args.seed)
    use_wandb = args.wandb
    if not use_wandb:
        print("[WARNING] Please note that you are not using wandb right now!!!")

    project_name = "haco_2022"
    team_name = "drivingforce"

    trial_name = "{}_seed{}_{}".format(exp_name, seed, get_time_str())
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
            "controller": "joystick",  # ========== Change to "joystick"!
            "window_size": (1600, 1100),
            "cos_similarity": True,
            # "map": "COT",
            # "environment_num": 1,

            # "force_render_fps": 10,
        },

        # Algorithm config
        algo=dict(
            policy=HACOPolicy,
            replay_buffer_class=HACOReplayBuffer,  ###
            replay_buffer_kwargs=dict(
                discard_reward=True,  # PZH: We run in reward-free manner!
                discard_takeover_start=False,
                takeover_stop_td=False
            ),
            policy_kwargs=dict(
                share_features_extractor=False,  # PZH: Using independent CNNs for actor and critics
                # net_arch=[256, ]
            ),

            env=None,
            learning_rate=dict(
                actor=3e-4,
                critic=3e-4,
                entropy=3e-4,
            ),

            optimize_memory_usage=True,

            buffer_size=50_000,  # We only conduct experiment less than 50K steps

            learning_starts=100,  ###
            batch_size=128,  # Reduce the batch size for real-time copilot
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,

            action_noise=None,
            ent_coef="auto",
            target_update_interval=1,
            target_entropy="auto",
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
            save_freq=200,
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
    model = HACO(**config["algo"])

    # ===== Launch training =====
    model.learn(
        # training
        total_timesteps=100_0000,
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
