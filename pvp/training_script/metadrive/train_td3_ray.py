import copy
from ray import tune

from pvp_iclr_release.utils.expert_common import SaverCallbacks
from pvp_iclr_release.utils.expert_human_in_the_loop_env import HumanInTheLoopEnv
from pvp_iclr_release.utils.train_eval_config import baseline_eval_config
from pvp_iclr_release.utils.rllib_utils import get_train_parser
from pvp_iclr_release.utils.rllib_utils.train import train
from ray.rllib.agents.ddpg.td3 import TD3Trainer

evaluation_config = {"env_config": copy.deepcopy(baseline_eval_config)}

if __name__ == '__main__':
    args = get_train_parser().parse_args()

    exp_name = args.exp_name or "TD3"
    stop = {"timesteps_total": 200_0000}

    config = dict(
        env=HumanInTheLoopEnv,
        env_config=dict(
            main_exp=False
        ),

        # ===== Evaluation =====
        evaluation_interval=1,
        evaluation_num_episodes=30,
        evaluation_config=evaluation_config,
        evaluation_num_workers=2,
        metrics_smoothing_episodes=30,

        # ===== Training =====
        actor_hiddens=[256, 256],
        critic_hiddens=[256, 256],
        actor_lr=1e-4,
        critic_lr=1e-4,
        prioritized_replay=tune.grid_search([False, True]),
        horizon=1500,
        # target_network_update_freq=0,
        train_batch_size=100,
        timesteps_per_iteration=1000,
        learning_starts=10000,
        clip_actions=False,
        clip_rewards=tune.grid_search([True, False]),
        policy_delay=tune.grid_search([0, 1]),
        normalize_actions=True,
        num_cpus_for_driver=0.5,
        # No extra worker used for learning. But this config impact the evaluation workers.
        num_cpus_per_worker=0.1,
        # num_gpus_per_worker=0.1 if args.num_gpus != 0 else 0,
        num_gpus=0.1 if args.num_gpus != 0 else 0
    )

    train(
        TD3Trainer,
        exp_name=exp_name,
        keep_checkpoints_num=5,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        # num_seeds=2,
        num_seeds=3,
        custom_callback=SaverCallbacks,
        # test_mode=True,
        # local_mode=True

        wandb_key_file="~/wandb_api_key_file.txt",
        wandb_project="old_2022",

    )
