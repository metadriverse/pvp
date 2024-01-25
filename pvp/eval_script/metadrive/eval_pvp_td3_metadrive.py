import argparse
from pvp.eval_script.metadrive.eval_metadrive_utils import evaluate_metadrive_once
import os.path as osp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=str)
    parser.add_argument("--ret_save_folder", required=True, type=str)
    parser.add_argument("--start_ckpt", required=True, type=int)
    parser.add_argument("--num_ckpt", type=int, default=20)
    parser.add_argument("--skip", type=int, default=200)
    parser.add_argument("--num_ep_in_one_env", type=int, default=5)
    parser.add_argument("--total_env_num", type=int, default=50)
    args = parser.parse_args()

    use_render = False

    trial_path = args.path

    start_ckpt = args.start_ckpt
    num_ckpt = args.num_ckpt

    for ckpt_index in reversed(range(start_ckpt, start_ckpt + num_ckpt, args.skip)):
        ckpt_path = osp.join(trial_path, "rl_model_{}_steps.zip".format(ckpt_index))
        if not osp.exists(ckpt_path):
            print("=====\nWe can't find checkpoint {}\n=====".format(ckpt_path))
            continue

        print(
            "===== Start evaluating checkpoint {}. Will be saved at {} =====".format(ckpt_index, args.ret_save_folder)
        )
        ret = evaluate_metadrive_once(
            ckpt_path=args.path,
            ckpt_index=ckpt_index,
            folder_name=args.ret_save_folder,
            use_render=False,
            num_ep_in_one_env=args.num_ep_in_one_env,
            total_env_num=args.total_env_num,
        )
        if ret is None:
            print("We failed to evaluate.")
        else:
            print("\n\n\n Finish evaluation. \n\n\n")
