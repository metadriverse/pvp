# training script for old old and pvp for carlas, change line 66 and 150 to switchss
import argparse
import os
import os.path as osp

from pvp_iclr_release.utils.carla.pvp_carla_env import PVPEnv
from pvp_iclr_release.stable_baseline3.common.callbacks import CallbackList, CheckpointCallback
from pvp_iclr_release.stable_baseline3.common.monitor import Monitor
from pvp_iclr_release.stable_baseline3.common.wandb_callback import WandbCallback
from pvp_iclr_release.pvp.pvp_td3.pvp_td3 import pvpTD3
from pvp_iclr_release.stable_baseline3.td3.policies import TD3Policy
from pvp_iclr_release.stable_baseline3.old.old_buffer import oldReplayBuffer
from pvp_iclr_release.stable_baseline3.sac.our_features_extractor import OurFeaturesExtractor
from pvp_iclr_release.utils.older_utils import get_time_str
from pvp_iclr_release.stable_baseline3.old import oldPolicy, oldReplayBuffer, old
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", default="CARLA-OLDold", type=str, help="The experiment name.")
    parser.add_argument("--port", default=9000, type=int, help="Carla server port.")
    parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    parser.add_argument("--obs-mode", default="birdview", choices=["birdview", "first", "birdview42", "firststack"],
                        help="Set to True to upload stats to wandb.")
    args = parser.parse_args()

    # ===== Setup some meta information =====
    exp_name = args.exp_name
    port = args.port
    seed = int(args.seed)
    use_wandb = args.wandb
    if not use_wandb:
        print("[WARNING] Please note that you are not using wandb right now!!!")
    obs_mode = args.obs_mode

    project_name = "old_2022"
    team_name = "drivingforce"

    if obs_mode.endswith("stack"):
        other_feat_dim = 0
    else:
        other_feat_dim = 1

    trial_name = "{}_{}_seed{}_{}".format(exp_name, obs_mode, seed, get_time_str())
    log_dir = osp.join("runs", exp_name, trial_name)
    os.makedirs(osp.join("runs", exp_name), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    print("We start logging training data into {}".format(log_dir))

    # ===== Setup the config =====
    config = dict(

        # Environment config
        env_config=dict(
            obs_mode=obs_mode,
            force_fps=30,  ###
            disable_vis=False,  ###
            debug_vis=False,
            port=port,
            disable_takeover=False,  ###
            controller="keyboard",
            env={"visualize": {"location": "lower right"}}
        ),

        # Algorithm config
        algo=dict(
            policy=oldPolicy,
            replay_buffer_class=oldReplayBuffer,  ###
            replay_buffer_kwargs=dict(
                discard_reward=True  # xxx: We run in reward-free manner!
            ),
            policy_kwargs=dict(
                features_extractor_class=OurFeaturesExtractor,
                features_extractor_kwargs=dict(
                    features_dim=256 + other_feat_dim
                ),
                share_features_extractor=False,  # xxx: Using independent CNNs for actor and critics
                net_arch=[256, ]
            ),

            env=None,
            learning_rate=dict(
                actor=3e-4,
                critic=3e-4,
                entropy=3e-4,
            ),
            # learning_rate=1e-4,

            optimize_memory_usage=True,old

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
        port=port,
        seed=seed,
        use_wandb=use_wandb,
        obs_mode=obs_mode,
        trial_name=trial_name,
        log_dir=log_dir
    )

    # ===== Setup the training environment =====
    train_env = PVPEnv(config=config["env_config"], )
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
    model = old(**config["algo"])

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
