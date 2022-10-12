import os

from pvp_iclr_release.stable_baseline3.a2c import A2C
from pvp_iclr_release.stable_baseline3.common.utils import get_system_info
from pvp_iclr_release.stable_baseline3.ddpg import DDPG
from pvp_iclr_release.stable_baseline3.dqn import DQN
from pvp_iclr_release.stable_baseline3.her.her_replay_buffer import HerReplayBuffer
from pvp_iclr_release.stable_baseline3.ppo import PPO
from pvp_iclr_release.stable_baseline3.sac import SAC
from pvp_iclr_release.stable_baseline3.td3 import TD3

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file, "r") as file_handler:
    __version__ = file_handler.read().strip()


def HER(*args, **kwargs):
    raise ImportError(
        "Since Stable Baselines 2.1.0, `HER` is now a replay buffer class `HerReplayBuffer`.\n "
        "Please check the documentation for more information: https://stable-baselines3.readthedocs.io/"
    )
