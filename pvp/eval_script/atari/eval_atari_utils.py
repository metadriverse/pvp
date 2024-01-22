import os
import time

import gym
import numpy as np
import pandas as pd

from pvp.sb3.common.atari_wrappers import AtariWrapper
from pvp.sb3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from pvp.sb3.dqn.policies import CnnPolicy
# from pvp.utils.expert_human_in_the_loop_env import HumanInTheLoopEnv
from pvp.utils.print_dict_utils import pretty_print
from pvp.pvp.pvp_dqn.pvp_dqn import pvpDQN
from pvp.training_script.atari.train_atari_dqn import DQN
from pvp.sb3.sac.our_features_extractor import OurFeaturesExtractor
from pvp.utils.atari.atari_env_wrapper import HumanInTheLoopAtariWrapper

EVAL_ENV_START = 0


class AtariPolicyFunction:
    def __init__(self, ckpt_path, ckpt_index, env):
        self.algo = DQN(
            policy=CnnPolicy,
            policy_kwargs=dict(
                features_extractor_class=OurFeaturesExtractor,
                features_extractor_kwargs=dict(features_dim=256),
                # share_features_extractor=False,  # xxx: Using independent CNNs for actor and critics
                net_arch=[
                    256,
                ]
            ),
            env=env,
            learning_rate=1e-4,
            optimize_memory_usage=True,
            buffer_size=100000,
            batch_size=2,  # Reduce the batch size for real-time copilot
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            # tensorboard_log=log_dir,
            create_eval_env=False,
            verbose=2,
            # seed=0,
            device="auto",
            learning_starts=0,
            exploration_fraction=0.0,
            exploration_final_eps=0.0,
        )

        self.algo.set_parameters(load_path_or_dict=ckpt_path + "rl_model_{}_steps".format(ckpt_index))

    def __call__(self, o, deterministic=False):
        assert deterministic
        action, state = self.algo.predict(o, deterministic=deterministic)
        return action


def evaluate_atari_once(
    ckpt_path,
    ckpt_index,
    folder_name,
    use_render=False,
    num_ep_in_one_env=5,
    env_name="BreakoutNoFrameskip-v4",
):
    ckpt_name = "checkpoint_{}".format(ckpt_index)
    # ===== Evaluate populations =====
    os.makedirs("evaluate_results", exist_ok=True)
    saved_results = []

    from pvp.sb3.common.monitor import Monitor
    from gym.wrappers.time_limit import TimeLimit

    def _make_env():
        env = gym.make(env_name)
        env = TimeLimit(env, max_episode_steps=10_000)

        # xxx: Monitor is extremely important! We must have it in evaluation!
        env = Monitor(env=env, filename=None)
        env = AtariWrapper(env=env)

        # xxx: Human wrapper is used as a helper for auto-visualization!
        # env = HumanInTheLoopAtariWrapper(env, enable_human=False, enable_render=False, mock_human_behavior=False)
        return env

    env = VecFrameStack(DummyVecEnv([_make_env for _ in range(10)]), n_stack=4)

    # Setup policy
    # try:
    policy_function = AtariPolicyFunction(ckpt_path, ckpt_index, env)
    # except FileNotFoundError:
    #     print("We failed to load: ", ckpt_path)
    #     return None

    os.makedirs(folder_name, exist_ok=True)

    # Setup environment

    try:
        need_break = False
        # start = time.time()
        # last_time = time.time()
        ep_count = 0
        step_count = [0 for _ in range(env.num_envs)]
        # rewards = 0
        # ep_times = []

        # env_index = 0
        o = env.reset()

        # num_ep_in = 0

        while not need_break:
            # INPUT: [batch_size, obs_dim] or [obs_dim, ] array.
            # OUTPUT: [batch_size, act_dim] !! This is important!
            action = policy_function(o, deterministic=True)

            # Step the environment
            o, _, dones, infos = env.step(action)
            # rewards += r[0]

            # info = info[0]
            # d = d[0]

            for env_id, info in enumerate(infos):

                step_count[env_id] += 1

                if use_render:
                    env.render()

                if step_count[env_id] % 1000 == 0:
                    print("Step {}, Episode {} ({})".format(step_count[env_id], ep_count, num_ep_in_one_env))

                # Reset the environment
                # xxx: Can't use d=True as criterion!
                # if d or (step_count >= 10000):
                if dones[env_id]:
                    print("Env {} finish one 'life'!".format(env_id))

                if "episode" in info:

                    # xxx: This calculation is wrong! Use Monitor instead!!
                    # res = dict(episode_reward=rewards, episode_length=step_count)

                    assert "episode" in info, "You should use Monitor wrapper!"
                    episode_reward = info["episode"]["r"]
                    episode_length = info["episode"]["l"]

                    res = dict(episode_reward=episode_reward, episode_length=episode_length)

                    # print(
                    #     "Num episodes: {} ({}), Num steps in this episode: {}, "
                    #     "Ckpt: {}".format(
                    #         ep_count, num_ep_in_one_env, step_count,
                    #         # np.mean(ep_times), time.time() - start,
                    #         ckpt_path
                    #     )
                    # )
                    step_count[env_id] = 0
                    # rewards = 0
                    ep_count += 1
                    # num_ep_in += 1
                    # env_id_recorded = EVAL_ENV_START + env_index
                    # num_ep_in_recorded = num_ep_in

                    # xxx: Vectorized environment don't need reset!
                    # o = env.reset()

                    # ep_times.append(time.time() - last_time)
                    # last_time = time.time()

                    # print("Finish {} episodes with {:.3f} s!".format(ep_count, time.time() - start))
                    # res = env.get_episode_result()
                    res["episode"] = ep_count
                    res["ckpt_index"] = ckpt_index
                    res["env_id"] = env_id
                    # res["env_id"] = env_id_recorded
                    # res["num_ep_in_one_env"] = num_ep_in_recorded
                    saved_results.append(res)
                    df = pd.DataFrame(saved_results)

                    print("=== Episode {} ({}) Report ===".format(ep_count, num_ep_in_one_env))

                    path = "{}/{}_tmp.csv".format(folder_name, ckpt_name)
                    print("Backup data is saved at: ", path)
                    df.to_csv(path)

                    print(pretty_print(res))

                    if ep_count >= num_ep_in_one_env:
                        need_break = True

    except Exception as e:
        raise e
    finally:
        env.close()

    df = pd.DataFrame(saved_results)
    print("===== Result =====")
    print(
        "Checkpoint {} results (Total {} episodes): \n{}".format(
            ckpt_name, len(df), {k: np.round(df[k].mean(), 3)
                                 for k in df}
        )
    )
    print("===== Result =====")
    path = "{}/{}.csv".format(folder_name, ckpt_name)
    print("Final data is saved at: ", path)
    df.to_csv(path)
    df["model_name"] = ckpt_name
    return df


if __name__ == '__main__':
    ret = evaluate_atari_once(
        ckpt_path=os.path.expanduser("/home/xxx/model/ski/"),
        ckpt_index=20600,
        folder_name="test_eval",
        use_render=False,
        num_ep_in_one_env=50,
        env_name="SkiingNoFrameskip-v4",
    )
    if ret is None:
        print("We failed to evaluate.")
    else:
        print("\n\n\n Finish evaluation. \n\n\n")
