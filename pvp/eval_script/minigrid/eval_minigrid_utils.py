#script for eval a ckpt for minigrid
import os
import time
import os.path as osp
import gym
import numpy as np
import pandas as pd
import torch
from gym_minigrid.wrappers import ImgObsWrapper
from pvp.sb3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from pvp.sb3.dqn.policies import CnnPolicy
from pvp.utils.print_dict_utils import pretty_print
from pvp.pvp.pvp_dqn.pvp_dqn import pvpDQN
from pvp.training_script.minigrid.minigrid_env import MinigridWrapper
from pvp.training_script.minigrid.minigrid_model import MinigridCNN

EVAL_ENV_START = 0


class AtariPolicyFunction:
    def __init__(self, ckpt_path, ckpt_index, env):
        self.algo = oldDQN(
            policy=CnnPolicy,
            policy_kwargs=dict(features_extractor_class=MinigridCNN, activation_fn=torch.nn.Tanh, net_arch=[
                64,
            ]),
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
            # verbose=2,
            # seed=0,
            device="auto",
            learning_starts=0,
            exploration_fraction=0.0,
            exploration_final_eps=0.0,
        )

        self.algo.set_parameters(load_path_or_dict=ckpt_path + "/rl_model_{}_steps".format(ckpt_index))

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

    eval_log_dir = osp.join(ckpt_path, "evaluations")

    def _make_eval_env():
        env = gym.make(env_name)
        env = MinigridWrapper(env, enable_render=False, enable_human=False)
        env = Monitor(env=env, filename=eval_log_dir)
        env = ImgObsWrapper(env)
        return env

    # eval_env = _make_eval_env()
    eval_env = VecFrameStack(DummyVecEnv([_make_eval_env]), n_stack=4)

    # Setup policy
    # try:
    policy_function = AtariPolicyFunction(ckpt_path, ckpt_index, eval_env)
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
        step_count = [0 for _ in range(eval_env.num_envs)]
        # rewards = 0
        # ep_times = []

        # env_index = 0
        o = eval_env.reset()

        # num_ep_in = 0

        while not need_break:
            # INPUT: [batch_size, obs_dim] or [obs_dim, ] array.
            # OUTPUT: [batch_size, act_dim] !! This is important!
            action = policy_function(o, deterministic=True)

            # Step the environment
            o, _, dones, infos = eval_env.step(action)
            # rewards += r[0]

            # info = info[0]
            # d = d[0]

            for env_id, info in enumerate(infos):

                step_count[env_id] += 1

                if use_render:
                    eval_env.render()

                if step_count[env_id] % 1000 == 0:
                    print("Step {}, Episode {} ({})".format(step_count[env_id], ep_count, num_ep_in_one_env))

                # Reset the environment
                # xxx: Can't use d=True as criterion!
                # if d or (step_count >= 10000):
                # if dones[env_id]:
                #     print("Env {} finish one 'life'!".format(env_id))

                if "episode" in info:

                    # xxx: This calculation is wrong! Use Monitor instead!!
                    # res = dict(episode_reward=rewards, episode_length=step_count)
                    # print("Working on Eposide: " + str(ep_count))
                    assert "episode" in info, "You should use Monitor wrapper!"
                    # episode_reward = info["episode"]["r"]
                    # episode_length = info["episode"]["l"]
                    episode_success = info['episode']['is_success']

                    res = dict(episode_success=episode_success)

                    step_count[env_id] = 0
                    # rewards = 0
                    ep_count += 1
                    res["episode"] = ep_count
                    res["ckpt_index"] = ckpt_index
                    res["env_id"] = env_id
                    saved_results.append(res)
                    df = pd.DataFrame(saved_results)

                    # print("=== Episode {} ({}) Report ===".format(ep_count, num_ep_in_one_env))
                    #
                    # path = "{}/{}_tmp.csv".format(folder_name, ckpt_name)
                    # print("Backup data is saved at: ", path)
                    # df.to_csv(path)

                    # print(pretty_print(res))

                    if ep_count >= num_ep_in_one_env:
                        need_break = True

    except Exception as e:
        raise e
    finally:
        eval_env.close()
    df = pd.DataFrame(saved_results)
    # print(df)
    print("===== Result =====")
    print(
        "Checkpoint {} results (Total {} episodes): \n{}".format(
            ckpt_name, len(df), {k: np.round(df[k].mean(), 3)
                                 for k in df}
        )
    )
    num_success = 0
    sucess_false_dict = df['episode_success'].value_counts().to_dict()
    if True in sucess_false_dict.keys():
        num_success = sucess_false_dict[True]

    print(
        "Episodes count: {}, Num success: {}, Success rate: {}".format(
            num_ep_in_one_env, num_success, num_success / num_ep_in_one_env
        )
    )
    print("===== Result =====")
    path = "{}/{}.csv".format(folder_name, ckpt_name)
    print("Final data is saved at: ", path)
    df.to_csv(path)
    df["model_name"] = ckpt_name
    return num_success / num_ep_in_one_env


if __name__ == '__main__':
    index_x = []
    success_y = []
    for ckpt_index in range(10, 1010, 10):
        success_rate = evaluate_atari_once(
            # ckpt_path=os.path.expanduser("/home/xxx/model/old_minigrid/runs/minigrid_old_2room/minigrid_old_2room_MiniGrid-MultiRoom-N2-S4-v0_lr0.0001_seed2_2022-06-08_23-20-31/models/"),
            ckpt_path=os.path.expanduser(
                "/home/xxx/model/old_minigrid/runs/minigrid_emptyroom/minigrid-emptyroom_MiniGrid-Empty-Random-6x6-v0_lr0.0001_seed2_2022-06-14_02-29-53/models/"
            ),
            ckpt_index=ckpt_index,
            folder_name=
            "/home/xxx/model/old_minigrid/runs/minigrid_emptyroom/minigrid-emptyroom_MiniGrid-Empty-Random-6x6-v0_lr0.0001_seed2_2022-06-14_02-29-53/test_eval",
            use_render=False,
            num_ep_in_one_env=50,
            # env_name="MiniGrid-MultiRoom-N2-S4-v0",
            env_name="MiniGrid-Empty-Random-6x6-v0",
            # env_name="MiniGrid-FourRooms-v0"
        )
        index_x.append(ckpt_index)
        success_y.append(success_rate)
    final_result = {'ckpt_index': index_x, "success_rate": success_y}
    resultdf = pd.DataFrame(final_result)
    resultdf.to_csv(
        "/home/xxx/model/old_minigrid/runs/minigrid_emptyroom/minigrid-emptyroom_MiniGrid-Empty-Random-6x6-v0_lr0.0001_seed2_2022-06-14_02-29-53/final_eval.csv"
    )
    import matplotlib.pyplot as plt
    plt.plot(index_x, success_y, 'o-', color='g')
    plt.xlabel("ckpt_index")
    plt.ylabel("success_rate")
    plt.show()
