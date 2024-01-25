#check multiple ckpt (success or not) for metadrive TD3 baseline
import copy
import os
import time

from pvp.utils.expert_human_in_the_loop_env import HumanInTheLoopEnv
from pvp.sb3.td3.policies import TD3Policy
from pvp.sb3.td3.td3 import TD3, ReplayBuffer
from pvp.utils.train_eval_config import baseline_eval_config
from pvp.utils.print_dict_utils import RecorderEnv

EVAL_ENV_START = baseline_eval_config["start_seed"]


class PolicyFunction:
    def __init__(self, ckpt_path, ckpt_index, env):
        self.algo = TD3(
            policy=TD3Policy,
            replay_buffer_class=ReplayBuffer,  ###
            replay_buffer_kwargs=dict(),
            policy_kwargs=dict(net_arch=[256, 256]),
            env=env,
            learning_rate=1e-4,
            optimize_memory_usage=True,
            learning_starts=10000,  ###
            batch_size=100,  # Reduce the batch size for real-time copilot
            tau=0.005,
            gamma=0.99,
            # train_freq=1,
            # target_policy_noise=0,
            # policy_delay=1,
            action_noise=None,
            create_eval_env=False,
            verbose=2,
            device="auto",
        )
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
    seed=3000,
):
    ckpt_name = "checkpoint_{}".format(ckpt_index)
    # ===== Evaluate populations =====
    os.makedirs("evaluate_results", exist_ok=True)
    saved_results = []
    env = make_metadrive_env(use_render, seed=seed)
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
        o = env.reset(force_seed=seed + env_index)

        num_ep_in = 0
        curr_success = False
        while True:
            # INPUT: [batch_size, obs_dim] or [obs_dim, ] array.
            # OUTPUT: [batch_size, act_dim] !! This is important!
            action = policy_function(o, deterministic=False)[0]

            o, r, d, info = env.step(action)
            step_count += 1
            # curr_name = "metadriveframe_{}.png".format('{0:05}'.format(step_count))
            # currimg = PNMImage()
            # env.engine.win.getScreenshot(currimg)
            # currimg.write(os.path.join(folder_name, curr_name))
            if use_render:
                env.render()
            # Reset the environment
            if d or (step_count >= 3000):
                step_count = 0
                ep_count += 1
                num_ep_in += 1
                if info['cost'] != 0 or not info['arrive_dest']:
                    print("Seed: " + str(seed) + "Failed!!!")
                    curr_success = True
                break
    except Exception as e:
        raise e
    finally:
        env.close()
    return curr_success


def make_metadrive_env(use_render=False, seed=3000):
    config = copy.deepcopy(baseline_eval_config)

    if use_render:
        config["use_render"] = True
        config["disable_model_compression"] = True
    config["start_seed"] = seed
    env = HumanInTheLoopEnv(config)
    return RecorderEnv(env)


if __name__ == '__main__':
    successseed = []
    for currseed in range(7000, 70000, 1000):
        print("=================================================================")
        print("Testing seed: " + str(currseed))
        ret = evaluate_metadrive_once(
            ckpt_path="/home/xxx/nvme/iclr_ckpt/metadrive_td3/",
            ckpt_index=2000000,
            folder_name="/home/xxx/nvme/iclr-visual/metadrive_td3",
            use_render=False,
            num_ep_in_one_env=5,
            total_env_num=50,
            seed=currseed,
        )
        if ret:
            successseed.append(currseed)
    print(successseed)
