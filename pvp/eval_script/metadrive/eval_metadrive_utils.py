import copy
import os
import time

import numpy as np
import pandas as pd
from pvp.utils.expert_human_in_the_loop_env import HumanInTheLoopEnv
from pvp.sb3.td3.policies import TD3Policy
from pvp.utils.train_eval_config import baseline_eval_config
from pvp.utils.print_dict_utils import pretty_print, RecorderEnv
from pvp.pvp_td3 import pvpTD3

EVAL_ENV_START = baseline_eval_config["start_seed"]


class PolicyFunction:
    def __init__(self, ckpt_path, ckpt_index, env):
        self.algo = pvpTD3(policy=TD3Policy, env=env, policy_kwargs=dict(net_arch=[256, 256]))
        self.algo.set_parameters(load_path_or_dict=ckpt_path + "/rl_model_{}_steps.zip".format(ckpt_index))

    def __call__(self, o, deterministic=False):
        return self.algo.predict(o, deterministic=deterministic)


def evaluate_metadrive_once(
    ckpt_path,
    ckpt_index,
    folder_name,
    use_render=False,
    num_ep_in_one_env=5,
    total_env_num=50,
):
    ckpt_name = "checkpoint_{}".format(ckpt_index)
    # ===== Evaluate populations =====
    os.makedirs("evaluate_results", exist_ok=True)
    saved_results = []
    env = make_metadrive_env(use_render)
    # Setup policy
    # try:
    policy_function = PolicyFunction(ckpt_path, ckpt_index, env)
    # except FileNotFoundError:
    #     print("We failed to load: ", ckpt_path)
    #     return None

    os.makedirs(folder_name, exist_ok=True)

    # Setup environment

    try:

        start = time.time()
        last_time = time.time()
        ep_count = 0
        step_count = 0
        ep_times = []

        env_index = 0
        o = env.reset(force_seed=EVAL_ENV_START + env_index)

        num_ep_in = 0

        while True:
            # INPUT: [batch_size, obs_dim] or [obs_dim, ] array.
            # OUTPUT: [batch_size, act_dim] !! This is important!
            action = policy_function(o, deterministic=False)[0]

            # Step the environment
            o, r, d, info = env.step(action)
            step_count += 1

            if use_render:
                env.render()

            # Reset the environment
            if d or (step_count >= 3000):

                print(
                    "Env {}, Num episodes: {} ({}), Num steps in this episode: {} (Ep time {:.2f}, "
                    "Total time {:.2f}). Ckpt: {}".format(
                        env_index, num_ep_in, ep_count, step_count, np.mean(ep_times),
                        time.time() - start, ckpt_path
                    )
                )
                step_count = 0
                ep_count += 1
                num_ep_in += 1
                env_id_recorded = EVAL_ENV_START + env_index
                num_ep_in_recorded = num_ep_in
                if num_ep_in >= num_ep_in_one_env:
                    env_index = min(env_index + 1, total_env_num - 1)
                    num_ep_in = 0

                o = env.reset(force_seed=EVAL_ENV_START + env_index)

                ep_times.append(time.time() - last_time)
                last_time = time.time()
                res = env.get_episode_result()
                print("Finish {} episodes with {:.3f} s!".format(ep_count, time.time() - start))
                res["episode"] = ep_count
                res["ckpt_index"] = ckpt_index
                res["env_id"] = env_id_recorded
                res["num_ep_in_one_env"] = num_ep_in_recorded
                saved_results.append(res)
                df = pd.DataFrame(saved_results)
                print(pretty_print(res))

                path = "{}/{}_tmp.csv".format(folder_name, ckpt_name)
                print("Backup data is saved at: ", path)
                df.to_csv(path)

                if env_index >= total_env_num - 1:
                    break

    except Exception as e:
        raise e
    finally:
        env.close()

    df = pd.DataFrame(saved_results)
    print("===== Result =====")
    print("Checkpoint {} results (len {}): \n{}".format(ckpt_name, len(df), {k: round(df[k].mean(), 3) for k in df}))
    print("===== Result =====")
    path = "{}/{}.csv".format(folder_name, ckpt_name)
    print("Final data is saved at: ", path)
    df.to_csv(path)
    df["model_name"] = ckpt_name
    return df


def make_metadrive_env(use_render=False):
    config = copy.deepcopy(baseline_eval_config)

    if use_render:
        config["use_render"] = True
        config["disable_model_compression"] = True

    env = HumanInTheLoopEnv(config)
    return RecorderEnv(env)


if __name__ == '__main__':
    ret = evaluate_metadrive_once(
        ckpt_path="C:\\Users\\78587\\Downloads\\",
        ckpt_index=40000,
        folder_name="test_eval",
        use_render=False,
        num_ep_in_one_env=5,
        total_env_num=50,
    )

    if ret is None:
        print("We failed to evaluate.")
    else:
        print("\n\n\n Finish evaluation. \n\n\n")
