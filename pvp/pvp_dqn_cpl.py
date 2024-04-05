import io
import pathlib
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union
from typing import Iterable
from typing import NamedTuple

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

import numpy as np
from pvp.sb3.haco.haco_buffer import concat_samples, HACOReplayBuffer
import torch
import torch as th
from gym import spaces
from gymnasium import spaces as new_spaces

from pvp.sb3.common.buffers import ReplayBuffer
from pvp.sb3.common.save_util import recursive_getattr, save_to_zip_file
from pvp.sb3.common.type_aliases import TensorDict
from pvp.sb3.common.vec_env import VecNormalize
from pvp.sb3.dqn.dqn import DQN


def biased_bce_with_logits(adv1, adv2, y, bias=1.0):
    # Apply the log-sum-exp trick.
    # y = 1 if we prefer x2 to x1
    # We need to implement the numerical stability trick.

    logit21 = adv2 - bias * adv1
    logit12 = adv1 - bias * adv2
    max21 = torch.clamp(-logit21, min=0, max=None)
    max12 = torch.clamp(-logit12, min=0, max=None)
    nlp21 = torch.log(torch.exp(-max21) + torch.exp(-logit21 - max21)) + max21
    nlp12 = torch.log(torch.exp(-max12) + torch.exp(-logit12 - max12)) + max12
    loss = y * nlp21 + (1 - y) * nlp12
    loss = loss.mean()

    # Now compute the accuracy
    with torch.no_grad():
        accuracy = ((adv2 > adv1) == torch.round(y)).float().mean()

    return loss, accuracy


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
    actions_behavior: th.Tensor
    actions_novice: th.Tensor


class HACOReplayBufferNew(ReplayBuffer):
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
        
        self.max_episode_length = 1000
        self.max_episodes = self.buffer_size // self.max_episode_length

        self.episode_pos = 0
        self.in_episode_pos = 0
        
        

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # PZH: We know support optimize_memory_usage!
        # assert optimize_memory_usage is False, "DictReplayBuffer does not support optimize_memory_usage"
        # disabling as this adds quite a bit of complexity
        # https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702
        self.optimize_memory_usage = optimize_memory_usage

        assert self.n_envs == 1

        self.observations = {
            # key: np.zeros((self.max_episodes, self.max_episode_length) + _obs_shape, dtype=self.observation_space[key].dtype)
            key: np.zeros((self.max_episodes, self.max_episode_length) + _obs_shape, dtype=self.observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }

        if self.optimize_memory_usage:
            self.next_observations = None
        else:
            self.next_observations = {
                key: np.zeros((self.max_episodes, self.max_episode_length) + _obs_shape, dtype=self.observation_space[key].dtype)
                for key, _obs_shape in self.obs_shape.items()
            }

        self.actions_behavior = np.zeros((self.max_episodes, self.max_episode_length, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.max_episodes, self.max_episode_length), dtype=np.float32)
        self.dones = np.zeros((self.max_episodes, self.max_episode_length), dtype=np.float32)

        # PZH: Add more buffers to store novice / expert actions and takeover
        self.interventions = np.zeros((self.max_episodes, self.max_episode_length), dtype=np.float32)
        self.intervention_starts = np.zeros((self.max_episodes, self.max_episode_length), dtype=np.float32)
        self.intervention_costs = np.zeros((self.max_episodes, self.max_episode_length), dtype=np.float32)
        self.actions_novice = np.zeros((self.max_episodes, self.max_episode_length, self.action_dim), dtype=action_space.dtype)
        self.discard_reward = discard_reward

        if not self.discard_reward:
            print("You are not discarding reward from the environment! This should be True when training HACO!")

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.max_episodes, self.max_episode_length), dtype=np.float32)

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
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])

            # self.observations[key][self.pos] = np.array(obs[key])
            self.observations[key][self.episode_pos][self.in_episode_pos] = np.array(obs[key])[0]

        for key in self.observations.keys():
            if isinstance(self.observation_space.spaces[key], (spaces.Discrete, new_spaces.Discrete)):
                next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            if self.optimize_memory_usage:
                # self.observations[key][(self.pos + 1) % self.buffer_size] = np.array(next_obs[key]).copy()
                self.observations[key][self.episode_pos][(self.in_episode_pos + 1) % self.max_episode_length] = np.array(next_obs[key]).copy()[0]
            else:
                raise ValueError()
                self.next_observations[key][self.pos] = np.array(next_obs[key]).copy()

        # self.dones[self.pos] = np.array(done).copy()
        self.dones[self.episode_pos][self.in_episode_pos] = np.array(done).copy()

        # PZH: Add useful data into buffers
        self.interventions[self.episode_pos][self.in_episode_pos] = np.array([step["takeover"] for step in infos])[0]
        self.intervention_starts[self.episode_pos][self.in_episode_pos] = np.array([step["takeover_start"] for step in infos])[0]
        self.intervention_costs[self.episode_pos][self.in_episode_pos] = np.array([step["takeover_cost"] for step in infos])[0]
        behavior_actions = np.array([step["raw_action"] for step in infos]).copy()
        if isinstance(self.action_space, (spaces.Discrete, new_spaces.Discrete)):
            action = action.reshape((self.n_envs, self.action_dim))
            behavior_actions = behavior_actions.reshape((self.n_envs, self.action_dim))
        self.actions_novice[self.episode_pos][self.in_episode_pos] = np.array(action).copy()[0]#.reshape(self.actions_novice[self.pos].shape)
        self.actions_behavior[self.episode_pos][self.in_episode_pos]  = behavior_actions[0]#.reshape(self.actions_behavior[self.pos].shape)
        if self.discard_reward:
            self.rewards[self.episode_pos][self.in_episode_pos]  = np.zeros_like(self.rewards[self.episode_pos][self.in_episode_pos] )
        else:
            self.rewards[self.episode_pos][self.in_episode_pos]  = np.array(reward).copy()[0]

        if self.handle_timeout_termination:
            self.timeouts[self.episode_pos][self.in_episode_pos]  = np.array([info.get("TimeLimit.truncated", False) for info in infos])[0]

        # self.pos += 1
        # if self.pos == self.buffer_size:
        #     self.full = True
        #     self.pos = 0

        self.in_episode_pos += 1
        if done[0]:
            self.episode_pos += 1
            self.in_episode_pos = 0
        if self.episode_pos == self.max_episodes:
            self.full = True
            self.episode_pos = 0


    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> HACODictReplayBufferSamples:
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)

        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        new_ret = self._get_samples(batch_inds, env=env)
        return new_ret

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> HACODictReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

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
            interventions=self.to_torch(self.interventions[batch_inds, env_indices].reshape(-1, 1), env),
            stop_td=self.to_torch(1 - _stop_td[batch_inds, env_indices].reshape(-1, 1), env),
            actions_behavior=self.to_torch(self.actions_behavior[batch_inds, env_indices]),
        )


class PVPDQNCPL(DQN):
    def __init__(self, q_value_bound=1., *args, **kwargs):
        kwargs["replay_buffer_class"] = HACOReplayBuffer
        if "replay_buffer_class" not in kwargs:
            kwargs["replay_buffer_class"] = HACOReplayBuffer
        super().__init__(*args, **kwargs)
        self.q_value_bound = q_value_bound

    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 4,
        eval_env=None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "run",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OffPolicyAlgorithm":

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

        callback.on_training_end()

        return self

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:

        if self.replay_buffer.pos == 0:
        # if self.replay_buffer.episode_pos== 0:
            return

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)
        stat_recorder = defaultdict(list)

        bc_loss_weight = 1.0  # TODO: Config

        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data_human = None
            replay_data_agent = None
            if self.replay_buffer.pos > batch_size and self.human_data_buffer.pos > batch_size:
                replay_data_agent = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
                replay_data_human = self.human_data_buffer.sample(batch_size, env=self._vec_normalize_env)
            elif self.human_data_buffer.pos > batch_size:
                replay_data_human = self.human_data_buffer.sample(batch_size, env=self._vec_normalize_env)
            elif self.replay_buffer.pos > batch_size:
                replay_data_agent = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            else:
                loss = None
                break

            alpha = 0.1  # TODO: Config
            accuracy = cpl_loss = bc_loss = None

            if replay_data_human is not None:
                human_action = replay_data_human.actions_behavior
                agent_action = replay_data_human.actions_novice
                policy_logit = self.policy.q_net(replay_data_human.observations)
                dist = torch.distributions.Categorical(logits=policy_logit)
                log_prob_human = dist.log_prob(human_action.flatten())#.sum(dim=-1)  # Don't do the sum...
                log_prob_agent = dist.log_prob(agent_action.flatten())#.sum(dim=-1)
                adv_human = alpha * log_prob_human
                adv_agent = alpha * log_prob_agent
                # If label = 1, then adv_human > adv_agent
                label = torch.ones_like(adv_human)
                cpl_loss, accuracy = biased_bce_with_logits(adv_agent, adv_human, label.float(), bias=0.5)

            if replay_data_agent is not None:
                # BC Loss for agent trajectory:
                policy_logit_bc = self.policy.q_net(replay_data_agent.observations)
                dist_bc = torch.distributions.Categorical(logits=policy_logit_bc)
                log_prob_bc = dist_bc.log_prob(replay_data_agent.actions_behavior.flatten())
                bc_loss = -log_prob_bc.mean()

            # Aggregate losses
            if bc_loss is None and cpl_loss is None:
                break

            loss = bc_loss_weight * (bc_loss if bc_loss is not None else 0.0) + (cpl_loss if cpl_loss is not None else 0.0)

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            # Stats
            stat_recorder["bc_loss"].append(bc_loss.item() if bc_loss is not None else float('nan'))
            stat_recorder["cpl_loss"].append(cpl_loss.item() if cpl_loss is not None else float('nan'))
            stat_recorder["cpl_accuracy"].append(accuracy.item() if accuracy is not None else float('nan'))

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", loss.item() if loss is not None else float('nan'))

        self.logger.record("train/agent_buffer_size", self.replay_buffer.get_buffer_size())
        # self.logger.record("train/human_buffer_size", self.human_data_buffer.get_buffer_size())

        for key in ['bc_loss', 'cpl_loss', 'cpl_accuracy']:
            self.logger.record("train/{}".format(key), np.mean(stat_recorder[key]))

        # Compute entropy (copied from RLlib TF dist)
        # self.logger.record("train/entropy", np.mean(entropies))

    def _setup_model(self) -> None:
        super()._setup_model()
        self.human_data_buffer = HACOReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            n_envs=self.n_envs,
            optimize_memory_usage=self.optimize_memory_usage,
            **self.replay_buffer_kwargs
        )
        # self.human_data_buffer = self.replay_buffer

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        if infos[0]["takeover"] or infos[0]["takeover_start"]:
            replay_buffer = self.human_data_buffer
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)

    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Save all the attributes of the object and the model parameters in a zip-file.

        :param path: path to the file where the rl agent should be saved
        :param exclude: name of parameters that should be excluded in addition to the default ones
        :param include: name of parameters that might be excluded but should be included anyway
        """
        # Copy parameter list so we don't mutate the original dict
        data = self.__dict__.copy()
        # print(data)
        del data["replay_buffer"]
        # del data["human_data_buffer"]
        # Exclude is union of specified parameters (if any) and standard exclusions
        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        # Do not exclude params if they are specifically included
        if include is not None:
            exclude = exclude.difference(include)

        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        all_pytorch_variables = state_dicts_names + torch_variable_names
        for torch_var in all_pytorch_variables:
            # We need to get only the name of the top most module as we'll remove that
            var_name = torch_var.split(".")[0]
            # Any params that are in the save vars must not be saved by data
            exclude.add(var_name)

        # Remove parameter entries of parameters which are to be excluded
        for param_name in exclude:
            data.pop(param_name, None)

        # Build dict of torch variables
        pytorch_variables = None
        if torch_variable_names is not None:
            pytorch_variables = {}
            for name in torch_variable_names:
                attr = recursive_getattr(self, name)
                pytorch_variables[name] = attr

        # Build dict of state_dicts
        params_to_save = self.get_parameters()

        save_to_zip_file(path, data=data, params=params_to_save, pytorch_variables=pytorch_variables)
