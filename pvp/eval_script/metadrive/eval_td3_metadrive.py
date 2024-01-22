import argparse
import os.path as osp

from pvp.eval_script.metadrive.eval_metadrive_utils import pretty_print, make_metadrive_env, PolicyFunction, os, time, \
    EVAL_ENV_START, np, pd


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--path", required=True, type=str)
    # parser.add_argument("--ret_save_folder", required=True, type=str)
    # parser.add_argument("--start_ckpt", required=True, type=int)
    # parser.add_argument("--num_ckpt", type=int, default=20)
    # parser.add_argument("--skip", type=int, default=200)
    parser.add_argument("--num_ep_in_one_env", type=int, default=2)
    parser.add_argument("--total_env_num", type=int, default=50)
    args = parser.parse_args()

    use_render = False

    # for ckpt_index in reversed(range(start_ckpt, start_ckpt + num_ckpt, args.skip)):
    # ckpt_path = osp.join(trial_path, "rl_model_{}_steps.zip".format(ckpt_index))
    ckpt_path = "/Users/pengzhenghao/Desktop/iclr 2022/Draw/metadrive_td3_eval/td3_metadrive_ckpt4/td3_metadrive025_sb3_seed0_2022-06-02_14-30-18"
    if not osp.exists(ckpt_path):
        print("=====\nWe can't find checkpoint {}\n=====".format(ckpt_path))
        raise ValueError()

    # print("===== Start evaluating checkpoint {}. Will be saved at {} =====".format(ckpt_index, args.ret_save_folder))
    ret = evaluate_metadrive_once(
        ckpt_path=ckpt_path,
        ckpt_index=2000000,
        folder_name="eval_ckpt4",
        use_render=False,
        num_ep_in_one_env=args.num_ep_in_one_env,
        total_env_num=args.total_env_num,
    )
    if ret is None:
        print("We failed to evaluate.")
    else:
        print("\n\n\n Finish evaluation. \n\n\n")
