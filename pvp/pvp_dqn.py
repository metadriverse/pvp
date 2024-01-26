import io
import pathlib
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import torch as th
from torch.nn import functional as F

from pvp.sb3.common.buffers import ReplayBuffer
from pvp.sb3.common.save_util import recursive_getattr, save_to_zip_file
from pvp.sb3.dqn.dqn import DQN, compute_entropy
from pvp.sb3.haco.haco_buffer import HACOReplayBuffer, concat_samples


class PVPDQN(DQN):
    def __init__(self, q_value_bound=1., *args, **kwargs):
        kwargs["replay_buffer_class"] = HACOReplayBuffer
        if "replay_buffer_class" not in kwargs:
            kwargs["replay_buffer_class"] = HACOReplayBuffer
        super(PVPDQN, self).__init__(*args, **kwargs)
        self.q_value_bound = q_value_bound

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:

        if self.human_data_buffer.pos == 0 and self.replay_buffer.pos == 0:
            return

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)
        stat_recorder = defaultdict(list)

        losses = []
        entropies = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            if self.replay_buffer.pos > batch_size and self.human_data_buffer.pos > batch_size:
                replay_data_agent = self.replay_buffer.sample(int(batch_size / 2), env=self._vec_normalize_env)
                replay_data_human = self.human_data_buffer.sample(int(batch_size / 2), env=self._vec_normalize_env)
                replay_data = concat_samples(replay_data_agent, replay_data_human)
            elif self.human_data_buffer.pos > batch_size:
                replay_data = self.human_data_buffer.sample(batch_size, env=self._vec_normalize_env)
            elif self.replay_buffer.pos > batch_size:
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            else:
                break

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                assert replay_data.rewards.mean().item() == 0.0
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            entropies.append(compute_entropy(current_q_values))

            # Retrieve the q-values for the actions from the replay buffer
            current_behavior_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions_behavior.long())

            current_novice_q_value_method1 = th.gather(current_q_values, dim=1, index=replay_data.actions_novice.long())

            current_novice_q_value = current_novice_q_value_method1

            mask = replay_data.interventions
            stat_recorder["no_intervention_rate"].append(mask.float().mean().item())

            no_overlap = replay_data.actions_behavior != replay_data.actions_novice
            stat_recorder["no_overlap_rate"].append(no_overlap.float().mean().item())
            stat_recorder["masked_no_overlap_rate"].append((mask * no_overlap).float().mean().item())

            pvp_loss = \
                F.mse_loss(
                    mask * current_behavior_q_values,
                    mask * self.q_value_bound * th.ones_like(current_behavior_q_values)
                ) + \
                F.mse_loss(
                    mask * no_overlap * current_novice_q_value,
                    mask * no_overlap * (-self.q_value_bound) * th.ones_like(current_novice_q_value)
                )

            # Compute Huber loss (less sensitive to outliers)
            loss_td = F.smooth_l1_loss(current_behavior_q_values, target_q_values)

            loss = loss_td.mean() + pvp_loss.mean()
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            stat_recorder["q_value_behavior"].append(current_behavior_q_values.mean().item())
            stat_recorder["q_value_novice"].append(current_novice_q_value.mean().item())
        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

        self.logger.record("train/agent_buffer_size", self.replay_buffer.get_buffer_size())
        self.logger.record("train/human_buffer_size", self.human_data_buffer.get_buffer_size())

        for key, values in stat_recorder.items():
            self.logger.record("train/{}".format(key), np.mean(values))

        # Compute entropy (copied from RLlib TF dist)
        self.logger.record("train/entropy", np.mean(entropies))

    def _setup_model(self) -> None:
        super(PVPDQN, self)._setup_model()
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
        super(PVPDQN, self)._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)

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
        del data["human_data_buffer"]
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
