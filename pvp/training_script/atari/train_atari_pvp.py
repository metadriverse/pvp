import argparse
import os
import os.path as osp

import gym

from pvp_iclr_release.stable_baseline3.common.atari_wrappers import AtariWrapper
from pvp_iclr_release.stable_baseline3.common.callbacks import CallbackList, CheckpointCallback
from pvp_iclr_release.stable_baseline3.common.monitor import Monitor
from pvp_iclr_release.stable_baseline3.common.vec_env import DummyVecEnv, VecFrameStack
from pvp_iclr_release.stable_baseline3.common.wandb_callback import WandbCallback
from pvp_iclr_release.stable_baseline3.dqn.policies import CnnPolicy
from pvp_iclr_release.utils.older_utils import get_time_str
from pvp_iclr_release.pvp.pvp_dqn.pvp_dqn import pvpDQN
from pvp_iclr_release.utils.atari.atari_env_wrapper import HumanInTheLoopAtariWrapper
from pvp_iclr_release.stable_baseline3.sac.our_features_extractor import OurFeaturesExtractor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", default="Fixed_Discrete_PVP", type=str, help="The experiment name.")
    parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    parser.add_argument("--obs-mode", default="image", choices=["vector", "image"],
                        help="Set to True to upload stats to wandb.")
    parser.add_argument("--env-name", default="BreakoutNoFrameskip-v4", type=str, help="Name of Gym environment")
    args = parser.parse_args()

    # ===== Setup some meta information =====
    exp_name = args.exp_name
    seed = int(args.seed)
    use_wandb = args.wandb
    if not use_wandb:
        print("[WARNING] Please note that you are not using wandb right now!!!")
    obs_mode = args.obs_mode
    env_name = args.env_name


    # if obs_mode != "vector":
    #     raise ValueError("we current only support state obs in vector")

    trial_name = "{}_{}_{}_seed{}_{}".format(exp_name, env_name, obs_mode, seed, get_time_str())
    log_dir = osp.join("runs", exp_name, trial_name)
    os.makedirs(osp.join("runs", exp_name), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    print("We start logging training data into {}".format(log_dir))

    # ===== Setup the config =====
    config = dict(

        # Environment config
        env_config=dict(),

        # Algorithm config
        algo=dict(
            policy=CnnPolicy,  # CNN observation
            # replay_buffer_class=oldReplayBuffer,  ###
            replay_buffer_kwargs=dict(
                discard_reward=True  # xxx: We run in reward-free manner!
            ),
            policy_kwargs=dict(
                features_extractor_class=OurFeaturesExtractor,
                features_extractor_kwargs=dict(
                    features_dim=256
                ),
                # share_features_extractor=False,  # xxx: Using independent CNNs for actor and critics
                net_arch=[256, ]
            ),

            exploration_initial_eps=.0,
            exploration_final_eps=.0,

            env=None,
            learning_rate=1e-4,

            optimize_memory_usage=True,
            buffer_size=100_000,  # We only conduct experiment less than 50K steps
            learning_starts=100,  ###
            batch_size=256,  # Reduce the batch size for real-time copilot
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=8,

            target_update_interval=1,
            tensorboard_log=log_dir,
            create_eval_env=False,

            verbose=2,
            seed=seed,
            device="auto",
        ),

        # Meta data
        # project_name=project_name,
        # team_name=team_name,
        exp_name=exp_name,
        seed=seed,
        use_wandb=use_wandb,
        obs_mode=obs_mode,
        trial_name=trial_name,
        log_dir=log_dir
    )

    # ===== Setup the training environment =====
    env = gym.make(env_name)
    env = Monitor(env=env, filename=log_dir)
    env = AtariWrapper(env=env)
    env = HumanInTheLoopAtariWrapper(env, enable_human=True, enable_render=True)
    train_env = VecFrameStack(DummyVecEnv([lambda: env]), n_stack=4)

    eval_log_dir = osp.join(log_dir, "evaluations")

    def _make_eval_env():
        env = gym.make(env_name)
        env = Monitor(env=env, filename=log_dir)
        env = AtariWrapper(env=env)
        # env = HumanInTheLoopAtariWrapper(env, enable_human=False, enable_render=False, mock_human_behavior=False)
        env = VecFrameStack(DummyVecEnv([lambda: env]), n_stack=4)
        return env

    eval_env = _make_eval_env()

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
    model = pvpDQN(**config["algo"])

    # ===== Launch training =====
    model.learn(
        # training
        total_timesteps=100_0000,
        callback=callbacks,
        reset_num_timesteps=True,

        # eval
        eval_env=eval_env,
        eval_freq=1000,
        n_eval_episodes=10,
        eval_log_path=None,

        # logging
        tb_log_name=exp_name,  # Should place the algorithm name here!
        log_interval=1,
    )
