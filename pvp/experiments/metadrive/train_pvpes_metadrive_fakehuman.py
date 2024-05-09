"""
Compared to original file:
1. use fakehumanenv
2. new config: free_level
3. buffer_size and total_timesteps set to 150_000
"""
import argparse
import os
from pathlib import Path
import uuid

from pvp.experiments.metadrive.egpo.fakehuman_env import FakeHumanEnv
from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
from pvp.pvp_td3 import PVPES
from pvp.sb3.common.callbacks import CallbackList, CheckpointCallback
from pvp.sb3.common.monitor import Monitor
from pvp.sb3.common.wandb_callback import WandbCallback
from pvp.sb3.haco import HACOReplayBuffer
from pvp.sb3.td3.policies import TD3Policy
from pvp.utils.shared_control_monitor import SharedControlMonitor
from pvp.utils.utils import get_time_str
from pvp.sb3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name", default="pvp_metadrive_fakehuman", type=str, help="The name for this batch of experiments."
    )
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
    parser.add_argument("--wandb_project", type=str, default="", help="The project name for wandb.")
    parser.add_argument("--wandb_team", type=str, default="", help="The team name for wandb.")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="/data/zhenghao/pvp", help="Folder to store the logs.")
    parser.add_argument("--free_level", type=float, default=0.95)
    parser.add_argument("--agent_data_ratio", type=float, default=1.0)
    parser.add_argument("--learning_starts", type=int, default=0)
    parser.add_argument("--eval_freq", type=int, default=500)
    parser.add_argument("--no_done_for_positive", type=str, default="False")
    parser.add_argument("--reward_0_for_positive", type=str, default="False")
    parser.add_argument("--reward_n2_for_intervention", type=str, default="False")
    parser.add_argument("--reward_1_for_all", type=str, default="False")
    parser.add_argument("--reward_0_for_negative", type=str, default="False")
    parser.add_argument("--use_weighted_reward", type=str, default="False")
    parser.add_argument("--remove_negative", type=str, default="False")
    parser.add_argument("--no_done_for_negative", type=str, default="False")
    parser.add_argument("--adaptive_batch_size", type=str, default="False")
    parser.add_argument("--add_bc_loss", type=str, default="False")

    parser.add_argument("--toy_env", action="store_true", help="Whether to use a toy environment.")
    parser.add_argument("--expert_deterministic", action="store_true")
    # parser.add_argument(
    #     "--device",
    #     required=True,
    #     choices=['wheel', 'gamepad', 'keyboard'],
    #     type=str,
    #     help="The control device, selected from [wheel, gamepad, keyboard]."
    # )
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
            expert_deterministic=args.expert_deterministic,
        ),

        # Algorithm config
        algo=dict(
            add_bc_loss=args.add_bc_loss,
            no_done_for_positive=args.no_done_for_positive,
            reward_0_for_positive=args.reward_0_for_positive,
            reward_n2_for_intervention=args.reward_n2_for_intervention,
            reward_1_for_all=args.reward_1_for_all,
            reward_0_for_negative=args.reward_0_for_negative,
            use_weighted_reward=args.use_weighted_reward,
            remove_negative=args.remove_negative,
            no_done_for_negative=args.no_done_for_negative,
            adaptive_batch_size=args.adaptive_batch_size,
            agent_data_ratio=args.agent_data_ratio,
            use_balance_sample=True,
            policy=TD3Policy,
            replay_buffer_class=HACOReplayBuffer,
            replay_buffer_kwargs=dict(
                discard_reward=True,  # We run in reward-free manner!
            ),
            policy_kwargs=dict(net_arch=[256, 256]),
            env=None,
            learning_rate=1e-4,
            q_value_bound=1,
            optimize_memory_usage=True,
            buffer_size=50_000,  # We only conduct experiment less than 50K steps
            learning_starts=args.learning_starts,  # The number of steps before
            batch_size=128,  # Reduce the batch size for real-time copilot
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
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
    train_env = SharedControlMonitor(env=train_env, folder=trial_dir / "data", prefix=trial_name)
    config["algo"]["env"] = train_env
    assert config["algo"]["env"] is not None

    # ===== Also build the eval env =====
    def _make_eval_env():
        eval_env_config = dict(
            use_render=False,  # Open the interface
            manual_control=False,  # Allow receiving control signal from external device
            start_seed=1000,
            horizon=1500,
        )
        from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
        from pvp.sb3.common.monitor import Monitor
        eval_env = HumanInTheLoopEnv(config=eval_env_config)
        eval_env = Monitor(env=eval_env, filename=str(trial_dir))
        return eval_env

    eval_env = SubprocVecEnv([_make_eval_env])

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
    if args.ckpt:
        model = PVPES.load(args.ckpt, **config["algo"])
    else:
        model = PVPES(**config["algo"])


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
        n_eval_episodes=50,
        eval_log_path=str(trial_dir),

        # logging
        tb_log_name=experiment_batch_name,
        log_interval=1,
        save_buffer=False,
        load_buffer=False,
    )
