# visualize (save image step by step) for atari pvp
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
        self.algo = pvpDQN(
            policy=CnnPolicy,
            policy_kwargs=dict(
                features_extractor_class=OurFeaturesExtractor,
                features_extractor_kwargs=dict(features_dim=256),
                # share_features_extractor=False,  # xxx: Using independent CNNs for actor and critics
                net_arch=[
                    256,
                ]
            ),
            exploration_initial_eps=.0,
            exploration_final_eps=.0,
            env=env,
            learning_rate=1e-4,
            optimize_memory_usage=True,
            buffer_size=100000,  # We only conduct experiment less than 50K steps
            learning_starts=100,  ###
            batch_size=256,  # Reduce the batch size for real-time copilot
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=8,
            target_update_interval=1,
            create_eval_env=False,
            verbose=2,
            # seed=seed,
            device="auto",
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
    env_name="SkiingNoFrameskip-v4",
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
        env.seed(70)

        # xxx: Human wrapper is used as a helper for auto-visualization!
        # env = HumanInTheLoopAtariWrapper(env, enable_human=False, enable_render=False, mock_human_behavior=False)
        return env

    # env = VecFrameStack(DummyVecEnv([_make_env for _ in range(10)]), n_stack=4)
    env = VecFrameStack(DummyVecEnv([_make_env]), n_stack=4)

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
            curr_name = "skiframe_{}.png".format('{0:03}'.format(step_count[0]))
            env.venv.envs[0].ale.saveScreenPNG(os.path.join(folder_name, curr_name))
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
                    print("Episode finished!")
                    need_break = True

    except Exception as e:
        raise e
    finally:
        env.close()
    return True


if __name__ == '__main__':
    ret = evaluate_atari_once(
        ckpt_path=os.path.expanduser(
            "/home/xxx/nvme/model/ski_formal/Fixed_Discrete_pvp_SkiingNoFrameskip-v4_image_seed0_2022-06-05_21-45-47/models/"
        ),
        ckpt_index=21000,
        folder_name="/home/xxx/nvme/iclr-visual/ski-pvp/",
        use_render=True,
        num_ep_in_one_env=1,
        env_name="SkiingNoFrameskip-v4",
    )
    if ret is None:
        print("We failed to evaluate.")
    else:
        print("\n\n\n Finish evaluation. \n\n\n")
