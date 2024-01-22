import argparse
import os.path as osp

from pvp.eval_script.atari.eval_atari_utils import evaluate_atari_once

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=str)
    parser.add_argument("--ret_save_folder", required=True, type=str)
    parser.add_argument("--start_ckpt", required=True, type=int)
    parser.add_argument("--num_ckpt", type=int, default=20)
    parser.add_argument("--skip", type=int, default=200)
    parser.add_argument("--num_ep_in_one_env", type=int, default=10)
    parser.add_argument("--env_name", type=int, default="BreakoutNoFrameskip-v4")
    args = parser.parse_args()

    use_render = False

    trial_path = args.path

    start_ckpt = args.start_ckpt
    num_ckpt = args.num_ckpt

    for ckpt_index in reversed(range(start_ckpt, start_ckpt + num_ckpt, args.skip)):
        ckpt_path = osp.join(trial_path, "checkpoint_{}".format(ckpt_index), "checkpoint-{}".format(ckpt_index))
        if not osp.exists(ckpt_path):
            print("=====\nWe can't find checkpoint {}\n=====".format(ckpt_path))
            continue

        print("===== Start evaluating checkpoint {}. Will be saved at {} =====".format(ckpt_index, args.folder))
        ret = evaluate_atari_once(
            ckpt_path=args.path,
            ckpt_index=ckpt_index,
            folder_name=args.ret_save_folder,
            use_render=False,
            num_ep_in_one_env=args.num_ep_in_one_env,
        )
        if ret is None:
            print("We failed to evaluate.")
        else:
            print("\n\n\n Finish evaluation. \n\n\n")
