import warnings
from typing import Any, Dict, List, Optional, Union
from typing import NamedTuple

import numpy as np
import torch as th
from gym import spaces
from gymnasium import spaces as new_spaces

from pvp.sb3.common.buffers import ReplayBuffer
from pvp.sb3.common.type_aliases import TensorDict
from pvp.sb3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class HACODictReplayBufferSamples(NamedTuple):
    observations: TensorDict
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor

    # PZH: Our new entries
    # actions: th.Tensor
    interventions: th.Tensor
    stop_td: th.Tensor
    intervention_costs: th.Tensor
    takeover_log_prob: th.Tensor
    actions_behavior: th.Tensor
    actions_novice: th.Tensor

    next_intervention_start: th.Tensor


def concat_samples(self, other):
    if isinstance(self.observations, dict):
        cat_obs = {k: th.concat([self.observations[k], other.observations[k]], dim=0) for k in self.observations.keys()}
        next_cat_obs = {
            k: th.concat([self.next_observations[k], other.next_observations[k]], dim=0)
            for k in self.next_observations.keys()
        }
    else:
        cat_obs = th.cat([self.observations, other.observations], dim=0)
        next_cat_obs = th.cat([self.next_observations, other.next_observations], dim=0)
    return HACODictReplayBufferSamples(
        cat_obs,
        next_cat_obs,
        dones=th.cat([self.dones, other.dones], dim=0),
        rewards=th.cat([self.rewards, other.rewards], dim=0),
        interventions=th.cat([self.interventions, other.interventions], dim=0),
        stop_td=th.cat([self.stop_td, other.stop_td], dim=0),
        intervention_costs=th.cat([self.intervention_costs, other.interventions], dim=0),
        takeover_log_prob=th.cat([self.takeover_log_prob, other.takeover_log_prob], dim=0),
        actions_behavior=th.cat([self.actions_behavior, other.actions_behavior], dim=0),
        actions_novice=th.cat([self.actions_novice, other.actions_novice], dim=0),
        next_intervention_start=th.cat([self.next_intervention_start, other.next_intervention_start], dim=0)
    )


class HACOReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = True,
        handle_timeout_termination: bool = True,
        discard_reward=False,
        discard_takeover_start=False,
        takeover_stop_td=False
    ):

        # Skip the init of ReplayBuffer and only run the BaseBuffer.__init__
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.discard_takeover_start = discard_takeover_start
        self.takeover_stop_td = takeover_stop_td
        # PZH: Hack
        # assert isinstance(self.obs_shape, dict), "DictReplayBuffer must be used with Dict obs space only"
        self._fake_dict_obs = False
        if not isinstance(self.obs_shape, dict):
            self.obs_shape = {"default": self.obs_shape}
            if isinstance(self.observation_space, spaces.Space):
                self.observation_space = spaces.Dict({'default': self.observation_space})
            elif isinstance(self.observation_space, new_spaces.Space):
                self.observation_space = new_spaces.Dict({'default': self.observation_space})
            else:
                raise ValueError("Unknown observation space {}".format(type(self.observation_space)))
            self._fake_dict_obs = True

        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # PZH: We know support optimize_memory_usage!
        # assert optimize_memory_usage is False, "DictReplayBuffer does not support optimize_memory_usage"
        # disabling as this adds quite a bit of complexity
        # https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = {
            key: np.zeros((self.buffer_size, self.n_envs) + _obs_shape, dtype=self.observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }

        if self.optimize_memory_usage:
            self.next_observations = None
        else:
            self.next_observations = {
                key: np.zeros((self.buffer_size, self.n_envs) + _obs_shape, dtype=self.observation_space[key].dtype)
                for key, _obs_shape in self.obs_shape.items()
            }

        self.actions_behavior = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # PZH: Add more buffers to store novice / expert actions and takeover
        self.interventions = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.intervention_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.intervention_costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.takeover_log_prob = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.actions_novice = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
        self.discard_reward = discard_reward

        if not self.discard_reward:
            print("You are not discarding reward from the environment! This should be True when training HACO!")

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            obs_nbytes = 0
            for _, obs in self.observations.items():
                obs_nbytes += obs.nbytes

            total_memory_usage = obs_nbytes + self.actions_behavior.nbytes + self.rewards.nbytes + self.dones.nbytes
            if self.next_observations is not None:
                next_obs_nbytes = 0
                for _, obs in self.observations.items():
                    next_obs_nbytes += obs.nbytes
                total_memory_usage += next_obs_nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:

        if infos[0]["takeover_start"] and self.discard_takeover_start:
            return

        if self._fake_dict_obs:
            obs = {"default": obs}
            next_obs = {"default": next_obs}

        # Copy to avoid modification by reference
        for key in self.observations.keys():
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], (spaces.Discrete, new_spaces.Discrete)):
                obs[key] = obs[key].reshape((self.n_envs, ) + self.obs_shape[key])
            self.observations[key][self.pos] = np.array(obs[key])

        for key in self.observations.keys():
            if isinstance(self.observation_space.spaces[key], (spaces.Discrete, new_spaces.Discrete)):
                next_obs[key] = next_obs[key].reshape((self.n_envs, ) + self.obs_shape[key])
            if self.optimize_memory_usage:
                self.observations[key][(self.pos + 1) % self.buffer_size] = np.array(next_obs[key]).copy()
            else:
                self.next_observations[key][self.pos] = np.array(next_obs[key]).copy()

        self.dones[self.pos] = np.array(done).copy()

        # PZH: Add useful data into buffers
        self.interventions[self.pos] = np.array([step["takeover"]
                                                 for step in infos]).reshape(self.interventions[self.pos].shape)
        self.intervention_starts[self.pos] = np.array([step["takeover_start"] for step in infos]
                                                      ).reshape(self.intervention_starts[self.pos].shape)
        self.intervention_costs[self.pos] = np.array([step["takeover_cost"] for step in infos]
                                                     ).reshape(self.intervention_costs[self.pos].shape)
        if "takeover_log_prob" in infos[0]:
            self.takeover_log_prob[self.pos] = np.array([step["takeover_log_prob"] for step in infos]
                                                        ).reshape(self.takeover_log_prob[self.pos].shape)
        else:
            self.takeover_log_prob[self.pos] = np.zeros_like(self.takeover_log_prob[self.pos])
        behavior_actions = np.array([step["raw_action"] for step in infos]).copy()
        if isinstance(self.action_space, (spaces.Discrete, new_spaces.Discrete)):
            action = action.reshape((self.n_envs, self.action_dim))
            behavior_actions = behavior_actions.reshape((self.n_envs, self.action_dim))
        self.actions_novice[self.pos] = np.array(action).copy().reshape(self.actions_novice[self.pos].shape)
        self.actions_behavior[self.pos] = behavior_actions.reshape(self.actions_behavior[self.pos].shape)

        # NOTE(PZH): a sanity check, if not takeover, then behavior actions must == novice actions
        # A special case is you might want to clip the novice actions.
        if not infos[0]["takeover"]:
            # TODO: This is overfit to MetaDrive, might need to fix.
            assert np.abs(np.clip(action, -1, 1) - behavior_actions).max() < 1e-6

        if self.discard_reward:
            self.rewards[self.pos] = np.zeros_like(self.rewards[self.pos])
        else:
            self.rewards[self.pos] = np.array(reward).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(
        self, batch_size: int, env: Optional[VecNormalize] = None, return_all=False
    ) -> HACODictReplayBufferSamples:
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super(HACOReplayBuffer, self).sample(batch_size=batch_size, env=env)

        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)

        if return_all:
            batch_inds = np.random.permutation(np.arange(self.buffer_size if self.full else self.pos))

        new_ret = self._get_samples(batch_inds, env=env)
        return new_ret

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> HACODictReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds), ))

        # Normalize if needed and remove extra dimension (we are using only one env for now)
        obs_ = self._normalize_obs(
            {key: obs[batch_inds, env_indices, :]
             for key, obs in self.observations.items()}, env
        )

        if not self.optimize_memory_usage:
            next_obs_ = self._normalize_obs(
                {key: obs[batch_inds, env_indices, :]
                 for key, obs in self.next_observations.items()}, env
            )
        else:
            next_obs_ = {}
            for key, obs in self.observations.items():
                next_obs_[key] = obs[(batch_inds + 1) % self.buffer_size, env_indices, :]
            next_obs_ = self._normalize_obs(next_obs_, env)

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        if self._fake_dict_obs:
            observations = observations["default"]
            next_observations = next_observations["default"]

        if self.takeover_stop_td:
            _stop_td = self.interventions
        else:
            _stop_td = self.intervention_starts

        next_intervention_start = self.intervention_starts[(batch_inds + 1) % self.buffer_size, env_indices]

        return HACODictReplayBufferSamples(
            observations=observations,
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])
                                ).reshape(-1, 1),
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),

            # PZH: Our useful data
            actions_novice=self.to_torch(self.actions_novice[batch_inds, env_indices]),
            intervention_costs=self.to_torch(self.intervention_costs[batch_inds, env_indices].reshape(-1, 1), env),
            takeover_log_prob=self.to_torch(self.takeover_log_prob[batch_inds, env_indices].reshape(-1, 1), env),
            interventions=self.to_torch(self.interventions[batch_inds, env_indices].reshape(-1, 1), env),
            stop_td=self.to_torch(1 - _stop_td[batch_inds, env_indices].reshape(-1, 1), env),
            actions_behavior=self.to_torch(self.actions_behavior[batch_inds, env_indices]),
            next_intervention_start=self.to_torch(next_intervention_start),
        )


class HACOReplayBufferEpisode(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,  # PZH: This is the number of episodes
        max_steps: int,  # PZH: This is the number of steps in each episode
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = True,
        handle_timeout_termination: bool = True,
        discard_reward=False,
    ):
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.max_steps = max_steps

        self.make_buffer = lambda: HACOReplayBuffer(
            buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage,
            handle_timeout_termination, discard_reward
        )
        self.episodes = [self.make_buffer()]

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        assert len(obs) == 1, "Only support one env for now"
        self.episodes[-1].add(obs, next_obs, action, reward, done, infos)
        if done[0]:
            # if not infos[0]['arrive_dest']:
            #     self.pos -= 1
            #     self.episodes = self.episodes[:-1]
            #     print("THIS EPISODE IS DISCARDED AS IT DOES NOT SUCCESS!!!! THIS IS DEBUG CODE AND SHOULD BE REMOVED!!!")
            #
            self.episodes.append(self.make_buffer())
            self.pos += 1

    def sample(
        self, batch_size: int, env: Optional[VecNormalize] = None, return_all=False,
            last_episodes = None
    ) -> HACODictReplayBufferSamples:
        """
        We will return everything we have!
        """
        if last_episodes is None:
            batch_inds = np.arange(self.buffer_size if self.full else self.pos)
        else:
            s = max(0, self.pos - last_episodes)
            e = self.pos
            batch_inds = np.arange(s, e)
        new_ret = self._get_samples(batch_inds, env=env)
        return new_ret

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None):
        """
        We will return everything we have!
        """
        ret = []
        for ep_count in batch_inds:
            ret.append(self.episodes[ep_count].sample(0, return_all=True))
        return ret
