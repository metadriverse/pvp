import argparse
import os
import os.path as osp
from collections import defaultdict

import pandas as pd

from pvp.pvp_td3 import pvpTD3
from pvp.experiments.carla.carla_env import HumanInTheLoopCARLAEnv
from pvp.sb3.sac.our_features_extractor import OurFeaturesExtractor
from pvp.sb3.td3.policies import TD3Policy
from pvp.sb3.old import oldPolicy, oldReplayBuffer, old
from pvp.sb3 import TD3
from pvp.sb3.td3.policies import MultiInputPolicy


def set_up_env(port, obs_mode, debug_vis=False, disable_vis=False, disable_takeover=True, force_fps=10):
    eval_env = HumanInTheLoopCARLAEnv(
        dict(
            obs_mode=obs_mode,
            force_fps=force_fps,
            disable_vis=disable_vis,
            debug_vis=debug_vis,
            port=port,
            disable_takeover=disable_takeover,
            show_text=False,
            normalize_obs=True,
            env={"visualize": {
                "location": "upper left"
            }}
        )
    )
    eval_env.seed(10)
    return eval_env


def setup_model_old(eval_env, seed, obs_mode):
    if obs_mode.endswith("stack"):
        other_feat_dim = 0
    else:
        other_feat_dim = 1

    # ===== Setup the config =====
    config = dict(
        # Algorithm config
        algo=dict(
            policy=oldPolicy,
            replay_buffer_class=oldReplayBuffer,  ###
            replay_buffer_kwargs=dict(discard_reward=True  # xxx: We run in reward-free manner!
                                      ),
            policy_kwargs=dict(
                features_extractor_class=OurFeaturesExtractor,
                features_extractor_kwargs=dict(features_dim=256 + other_feat_dim),
                share_features_extractor=False,  # xxx: Using independent CNNs for actor and critics
                net_arch=[
                    256,
                ]
            ),
            env=eval_env,
            learning_rate=dict(
                actor=3e-4,
                critic=3e-4,
                entropy=3e-4,
            ),
            # learning_rate=1e-4,
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
            # tensorboard_log=log_dir,
            create_eval_env=False,
            verbose=2,
            # seed=seed,
            device="auto",
        ),
    )

    # ===== Setup the training algorithm =====
    model = old(**config["algo"])
    return model


def setup_model(eval_env, seed, obs_mode):
    if obs_mode.endswith("stack"):
        other_feat_dim = 0
    else:
        other_feat_dim = 1

    # ===== Setup the config =====
    config = dict(
        # Algorithm config
        algo=dict(
            policy=TD3Policy,
            replay_buffer_class=oldReplayBuffer,  ###
            replay_buffer_kwargs=dict(discard_reward=True  # xxx: We run in reward-free manner!
                                      ),
            policy_kwargs=dict(
                features_extractor_class=OurFeaturesExtractor,
                features_extractor_kwargs=dict(features_dim=256 + other_feat_dim),
                share_features_extractor=False,  # xxx: Using independent CNNs for actor and critics
                net_arch=[
                    256,
                ]
            ),
            env=eval_env,
            # learning_rate=dict(
            #     actor=3e-4,
            #     critic=3e-4,
            #     entropy=3e-4,
            # ),
            optimize_memory_usage=True,
            buffer_size=10,  # 0.3e6
            learning_starts=10,  ###
            batch_size=220,  # Reduce the batch size for real-time copilot
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            action_noise=None,
            # ent_coef="auto",
            # target_update_interval=1,
            # target_entropy="auto",
            create_eval_env=False,
            verbose=2,
            seed=seed,
            device="auto",
        ),
    )

    # ===== Setup the training algorithm =====
    model = pvpTD3(**config["algo"])
    return model


def setup_model_td3(eval_env, seed, obs_mode):
    if obs_mode.endswith("stack"):
        other_feat_dim = 0
    else:
        other_feat_dim = 1

    # ===== Setup the config =====
    config = dict(
        # Algorithm config
        algo=dict(
            policy=MultiInputPolicy,
            policy_kwargs=dict(
                features_extractor_class=OurFeaturesExtractor,
                features_extractor_kwargs=dict(features_dim=256 + other_feat_dim, ),
                share_features_extractor=False,  # xxx: Using independent CNNs for actor and critics
                net_arch=[
                    256,
                ]
            ),
            env=eval_env,
            learning_rate=1e-4,
            buffer_size=500_000,  # 0.5e6
            learning_starts=10_000,
            batch_size=200,
            tau=0.005,
            gamma=0.99,
            # train_freq=1,
            # gradient_steps=1,
            action_noise=None,
            replay_buffer_class=None,
            replay_buffer_kwargs=None,
            # ent_coef="auto",
            # target_update_interval=1,
            # target_entropy="auto",
            # tensorboard_log=log_dir,
            create_eval_env=False,
            verbose=2,
            # seed=seed,
            device="auto",
        ),
    )

    # ===== Setup the training algorithm =====
    model = TD3(**config["algo"])
    return model


def update_model_if_needed(last_model, model, log_dir):

    possible_log_dirs = sorted(os.listdir(log_dir))
    if len(possible_log_dirs) == 0:
        print("No checkpoint found! Using initial policy now!")
        return last_model, model, None

    log_dir = possible_log_dirs[-1]
    print("=" * 60)
    print("We choose this log dir: ", log_dir)
    print("=" * 60)
    log_dir = osp.abspath(osp.join("runs", exp_name, log_dir))
    os.makedirs(osp.join(log_dir, "eval_results"), exist_ok=True)

    model_dir = osp.join(log_dir, "models")
    possible_models = sorted(os.listdir(model_dir), key=lambda v: int(v.split("_")[2]))
    print("We find the following checkpoints: ", possible_models)

    if len(possible_models) == 0:
        print("No checkpoint found! Using initial policy now!")
        return last_model, model, None

    model_path = possible_models[-1]
    if last_model != model_path:
        model_path = osp.join(model_dir, model_path)
        assert osp.isfile(model_path), model_path

        success_set = True
        try:
            model.set_parameters(model_path)
        except ValueError:
            success_set = False

        if success_set:
            print("=" * 60)
            print("Start evaluating the checkpoint: ", model_path)
            print("=" * 60)
            last_model = model_path

    return last_model, model, log_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", default="TEST", type=str, help="The experiment name.")
    parser.add_argument("--port", default=10000, type=int, help="Carla server port.")
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    parser.add_argument(
        "--obs-mode",
        default="birdview",
        choices=["birdview", "first", "birdview42", "firststack"],
        help="Set to True to upload stats to wandb."
    )
    args = parser.parse_args()

    exp_name = args.exp_name
    port = args.port
    seed = args.seed
    obs_mode = args.obs_mode

    exp_log_dir = osp.join("runs", exp_name)

    eval_env = set_up_env(port=args.port, obs_mode=obs_mode)
    model = setup_model(eval_env=eval_env, seed=seed, obs_mode=obs_mode)

    last_model = ""
    tmp_log_dir = None
    ep_count = 0
    recorder = defaultdict(list)
    obs = eval_env.reset()
    step_count = 0

    try:
        while True:
            if step_count % 100 == 0:
                last_model, model, tmp_log_dir = update_model_if_needed(last_model, model, exp_log_dir)

            action, _states = model.predict(obs, deterministic=False)
            obs, reward, done, info = eval_env.step(action)
            step_count += 1
            if done:
                ep_count += 1
                for k, v in info.items():
                    recorder[k].append(v)
                recorder["model"].append(last_model)
                recorder["ep_count"].append(ep_count)
                recorder["step_count"].append(step_count)

                print("The environment is terminated. Final info: ", info)
                obs = eval_env.reset()

                if tmp_log_dir is not None:
                    csv_file = osp.join(tmp_log_dir, "eval_results", "online_eval_result_{}.csv".format(ep_count))
                    df = pd.DataFrame(dict(recorder))
                    df.to_csv(csv_file)

    except KeyboardInterrupt:
        eval_env.close()
        del model
        print("Evaluation finished! Bye bye!")
