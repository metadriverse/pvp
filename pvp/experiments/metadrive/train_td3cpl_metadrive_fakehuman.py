import argparse
import os
from pathlib import Path
import uuid
import numpy as np


from pvp.experiments.metadrive.egpo.fakehuman_env import FakeHumanEnv
from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
from pvp.pvp_td3_cpl import PVPTD3CPL, PVPTD3CPLPolicy
from pvp.pvp_td3_cpl_real import PVPRealTD3Policy, PVPRealTD3CPL
from pvp.sb3.common.callbacks import CallbackList, CheckpointCallback
from pvp.sb3.common.monitor import Monitor
from pvp.sb3.common.wandb_callback import WandbCallback
from pvp.sb3.haco import HACOReplayBuffer
from pvp.sb3.td3.policies import TD3Policy
from pvp.utils.shared_control_monitor import SharedControlMonitor
from pvp.utils.utils import get_time_str
from pvp.sb3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from pvp.sb3.ppo.policies import MlpPolicy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name", default="td3cpl-metadrive-fake", type=str, help="The name for this batch of experiments."
    )
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
    parser.add_argument("--wandb_project", type=str, default="", help="The project name for wandb.")
    parser.add_argument("--wandb_team", type=str, default="", help="The team name for wandb.")
    parser.add_argument("--log_dir", type=str, default="/data/zhenghao/pvp", help="Folder to store the logs.")
    parser.add_argument("--free_level", type=float, default=0.95)

    parser.add_argument("--toy_env", action="store_true", help="Whether to use a toy environment.")
    # parser.add_argument(
    #     "--device",
    #     required=True,
    #     choices=['wheel', 'gamepad', 'keyboard'],
    #     type=str,
    #     help="The control device, selected from [wheel, gamepad, keyboard]."
    # )

    parser.add_argument("--prioritized_buffer", type=str, default="True")
    parser.add_argument("--use_chunk_adv", type=str, default="True")
    parser.add_argument("--training_deterministic", type=str, default="True")
    parser.add_argument("--add_loss_5", type=str, default="False")
    parser.add_argument("--add_loss_5_inverse", type=str, default="False")
    parser.add_argument("--mask_same_actions", type=str, default="False")
    parser.add_argument("--remove_loss_1", type=str, default="False")
    parser.add_argument("--remove_loss_3", type=str, default="False")
    parser.add_argument("--remove_loss_6", type=str, default="False")
    parser.add_argument("--add_bc_loss", type=str, default="False")
    parser.add_argument("--add_bc_loss_only_interventions", type=str, default="False")
    parser.add_argument("--use_target_policy", type=str, default="False")
    parser.add_argument("--use_target_policy_only_overwrite_takeover", type=str, default="False")
    parser.add_argument("--num_comparisons", type=int, default=64)
    parser.add_argument("--num_steps_per_chunk", type=int, default=64)
    parser.add_argument("--max_comparisons", type=int, default=10000)
    parser.add_argument("--n_eval_episodes", type=int, default=50)
    parser.add_argument("--eval_freq", type=int, default=500)
    parser.add_argument("--hard_reset", type=int, default=-1)
    parser.add_argument("--learning_starts", type=int, default=0)
    parser.add_argument("--cpl_bias", type=float, default=0.5)
    parser.add_argument("--top_factor", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--bc_loss_weight", type=float, default=-1.0)
    parser.add_argument("--last_ratio", type=float, default=-1)
    parser.add_argument("--log_std_init", type=float, default=0.0)

    parser.add_argument("--fixed_log_std", action="store_true")
    parser.add_argument("--eval_stochastic", action="store_true")
    parser.add_argument("--real_td3", action="store_true")
    parser.add_argument("--expert_deterministic", action="store_true")

    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--ckpt", type=str, default="")

    args = parser.parse_args()

    # ===== Set up some arguments =====
    # control_device = args.device
    experiment_batch_name = "{}_freelevel{}".format(args.exp_name, args.free_level)
    seed = args.seed
    trial_name = "{}_{}_{}".format(experiment_batch_name, get_time_str(), uuid.uuid4().hex[:8])
    print("Trial name is set to: ", trial_name)

    use_wandb = args.wandb
    project_name = args.wandb_project
    team_name = args.wandb_team
    if not use_wandb:
        print("[WARNING] Please note that you are not using wandb right now!!!")

    log_dir = args.log_dir
    experiment_dir = Path(log_dir) / Path("runs") / experiment_batch_name

    trial_dir = experiment_dir / trial_name
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(trial_dir, exist_ok=False)  # Avoid overwritting old experiment
    print(f"We start logging training data into {trial_dir}")

    free_level = args.free_level
    real_td3 = args.real_td3

    # TODO: CONFIG!!!!
    # def f(M):
    #     A = 2
    #     alpha = 0.1
    #     M = 1
    #     # log_std_init = (- M / (A * alpha)) - np.log(np.sqrt(2 * np.pi))
    #
    #     G = 2
    #     minus = A * alpha * G * G * np.pi / np.exp(-2*M/(A * alpha))
    #     min_val = M - minus
    #     print(min_val)
    #
    # def f(log_std):
    #     A = 2
    #     alpha = 0.1
    #     sigma = np.exp(log_std)
    #     G = 2
    #     max_a = - A * alpha * (log_std + np.log(np.sqrt(2 * np.pi)))
    #     min_a = - A * alpha * (G * G / (2 * sigma * sigma)) + max_a
    #     print(min_a, max_a)
    #     print(min_a * 64, max_a * 64)
    log_std_init = args.log_std_init

    # ===== Setup the config =====
    config = dict(

        # Environment config
        env_config=dict(

            # Original real human exp env config:
            # use_render=True,  # Open the interface
            # manual_control=True,  # Allow receiving control signal from external device
            # controller=control_device,
            # window_size=(1600, 1100),

            # FakeHumanEnv config:
            free_level=free_level,
            use_render=False,
            expert_deterministic=args.expert_deterministic,
        ),

        # Algorithm config
        algo=dict(

            use_chunk_adv=args.use_chunk_adv,
            num_comparisons=args.num_comparisons,
            num_steps_per_chunk=args.num_steps_per_chunk,
            prioritized_buffer=args.prioritized_buffer,
            training_deterministic=args.training_deterministic,
            add_bc_loss=args.add_bc_loss,
            cpl_bias=args.cpl_bias,
            add_loss_5=args.add_loss_5,
            add_loss_5_inverse=args.add_loss_5_inverse,
            top_factor=args.top_factor,
            mask_same_actions=args.mask_same_actions,
            remove_loss_1=args.remove_loss_1,
            remove_loss_3=args.remove_loss_3,
            remove_loss_6=args.remove_loss_6,
            hard_reset=args.hard_reset,
            use_target_policy=args.use_target_policy,
            last_ratio=args.last_ratio,
            max_comparisons=args.max_comparisons,
            use_target_policy_only_overwrite_takeover=args.use_target_policy_only_overwrite_takeover,
            bc_loss_weight=args.bc_loss_weight,
            add_bc_loss_only_interventions=args.add_bc_loss_only_interventions,

            use_balance_sample=True,
            policy=MlpPolicy if not real_td3 else PVPRealTD3Policy,
            replay_buffer_class=HACOReplayBuffer,  # TODO: USELESS
            replay_buffer_kwargs=dict(
                discard_reward=True,  # We run in reward-free manner!
                # max_steps=1000,  # TODO: CONFIG
            ),
            policy_kwargs=dict(
                net_arch=[256, 256],
                fixed_log_std=args.fixed_log_std,
                log_std_init=log_std_init,
            ),
            env=None,
            learning_rate=args.lr,

            # learning_rate=dict(
            #     actor=1e-4,
            #     critic=1e-4,
            #     entropy=1e-4,
            # ),
            q_value_bound=1,
            optimize_memory_usage=True,
            buffer_size=150_000,  # We only conduct experiment less than 50K steps
            learning_starts=args.learning_starts,  # The number of steps before
            batch_size=128,  # Reduce the batch size for real-time copilot
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "episode"),  # <<<<< This is very important
            gradient_steps=-1,  # <<<<< This is very important
            # gradient_steps=20,  # <<<<< This is very important
            action_noise=None,
            tensorboard_log=trial_dir,
            create_eval_env=False,
            verbose=2,
            seed=seed,
            device="auto",
        ),

        # Experiment log
        exp_name=experiment_batch_name,
        seed=seed,
        use_wandb=use_wandb,
        trial_name=trial_name,
        log_dir=str(trial_dir)
    )
    if args.toy_env:
        config["env_config"].update(
            # Here we set num_scenarios to 1, remove all traffic, and fix the map to be a very simple one.
            num_scenarios=1,
            traffic_density=0.0,
            map="COT"
        )

    # ===== Setup the training environment =====
    train_env = FakeHumanEnv(config=config["env_config"], )
    train_env = Monitor(env=train_env, filename=str(trial_dir))
    # Store all shared control data to the files.

    # TODO: FIXME:
    # TODO: FIXME:
    # TODO: FIXME:
    # TODO: FIXME: should add back when human experiemetn.
    # train_env = SharedControlMonitor(env=train_env, folder=trial_dir / "data", prefix=trial_name)


    config["algo"]["env"] = train_env
    assert config["algo"]["env"] is not None

    # ===== Also build the eval env =====
    def _make_eval_env(use_render=False):
        eval_env_config = dict(
            use_render=use_render,  # Open the interface
            manual_control=False,  # Allow receiving control signal from external device
            start_seed=1000,
            horizon=1500,


# start_seed=1024,
#             num_scenarios=1,
#             free_level=-1000,
#             expert_deterministic=True,
        )
        from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
        from pvp.sb3.common.monitor import Monitor

        eval_env = HumanInTheLoopEnv(config=eval_env_config)
        # eval_env = FakeHumanEnv(config=eval_env_config)

        eval_env = Monitor(env=eval_env, filename=str(trial_dir))
        return eval_env


    if args.eval:
        eval_env = SubprocVecEnv([lambda: _make_eval_env(True)])
        # eval_env = SubprocVecEnv([lambda: _make_eval_env(False)])
        config["algo"]["learning_rate"] = 0.0
        config["algo"]["train_freq"] = (1, "step")


        model = PVPTD3CPL.load(args.ckpt, **config["algo"])
        # model = PVPTD3CPL(**config["algo"])

        model.learn(
            # training
            total_timesteps=50_000,
            callback=None,
            reset_num_timesteps=True,

            # eval
            # eval_env=None,
            # eval_freq=-1,
            # n_eval_episodes=2,
            # eval_log_path=None,

            # eval
            eval_env=eval_env,
            eval_freq=1,
            n_eval_episodes=500,
            eval_log_path=str(trial_dir),

            # logging
            tb_log_name=experiment_batch_name,
            log_interval=1,
            save_buffer=False,
            load_buffer=False,

            eval_deterministic=not args.eval_stochastic,
        )
        exit(0)

    eval_env = SubprocVecEnv([_make_eval_env])
    # eval_env = None

    # ===== Setup the callbacks =====
    save_freq = 500  # Number of steps per model checkpoint
    callbacks = [
        CheckpointCallback(name_prefix="rl_model", verbose=1, save_freq=save_freq, save_path=str(trial_dir / "models"))
    ]
    if use_wandb:
        callbacks.append(
            WandbCallback(
                trial_name=trial_name,
                exp_name=experiment_batch_name,
                team_name=team_name,
                project_name=project_name,
                config=config
            )
        )
    callbacks = CallbackList(callbacks)

    # ===== Setup the training algorithm =====
    algo_cls = PVPRealTD3CPL if real_td3 else PVPTD3CPL
    if args.ckpt:
        from pvp.sb3.common.save_util import load_from_zip_file
        model = algo_cls(**config["algo"])
        data, params, pytorch_variables = load_from_zip_file(args.ckpt, device=model.device, print_system_info=False)
        model.set_parameters(params, exact_match=True, device=model.device)
    else:
        model = algo_cls(**config["algo"])

    # ===== Launch training =====
    model.learn(
        # training
        total_timesteps=50_000,
        callback=callbacks,
        reset_num_timesteps=True,

        # eval
        # eval_env=None,
        # eval_freq=-1,
        # n_eval_episodes=2,
        # eval_log_path=None,

        # eval
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        eval_log_path=str(trial_dir),

        # logging
        tb_log_name=experiment_batch_name,
        log_interval=1,
        save_buffer=False,
        load_buffer=False,
    )
