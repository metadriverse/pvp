import io
import os
import pathlib
from collections import defaultdict
from typing import Union, Dict, List, Any, Optional

import numpy as np
import torch
import torch as th
from torch.nn import functional as F

from pvp.pvp_td3_cpl import PVPTD3CPL, biased_bce_with_logits, log_probs_to_advantages, logger
from pvp.sb3.common.buffers import ReplayBuffer
from pvp.sb3.common.save_util import save_to_pkl, load_from_pkl
from pvp.sb3.common.type_aliases import MaybeCallback, GymEnv
from pvp.sb3.common.utils import polyak_update
from pvp.sb3.td3.policies import TD3Policy


class PVPRealTD3Policy(TD3Policy):
    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.reward_model = self.make_critic(features_extractor=None)
        self.reward_model.optimizer = self.optimizer_class(
            self.reward_model.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

        self.reward_model_target = self.make_critic(features_extractor=None)
        self.reward_model_target.load_state_dict(self.reward_model.state_dict())
        self.reward_model_target.set_training_mode(False)


class PVPRealTD3CPL(PVPTD3CPL):
    actor_update_count = 0

    def __init__(self, *args, **kwargs):
        for k in ["log_std_init", "fixed_log_std"]:
            if k in kwargs:
                kwargs.pop(k)
        super().__init__(*args, **kwargs)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

        self.reward_model = self.policy.reward_model
        self.reward_model_target = self.policy.reward_model_target

    def _setup_model(self) -> None:
        super()._setup_model()
        # if self.use_balance_sample:
        from pvp.sb3.haco.haco_buffer import HACOReplayBufferEpisode
        self.replay_buffer = HACOReplayBufferEpisode(
            buffer_size=self.buffer_size,
            max_steps=1000,  # TODO: CONFIG
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            n_envs=self.n_envs,
            optimize_memory_usage=self.optimize_memory_usage,
            **self.replay_buffer_kwargs
        )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        stat_recorder = defaultdict(list)

        # Sample replay buffer
        if self.replay_buffer.pos > 0:
            replay_data_agent = self.replay_buffer.sample(0, env=self._vec_normalize_env)
        else:
            return

        num_steps_per_chunk = self.extra_config["num_steps_per_chunk"]


        # if self.extra_config["use_chunk_adv"]:
        # Reorganize data with chunks
        # Now obs.shape = (#batches, #steps, #features)
        # We need to make it to be: obs.shape = (#batches, #steps-chunk_size, chunk_size, #features)
        new_obs = []
        new_action_behaviors = []
        new_action_novices = []

        new_valid_ep = []
        new_valid_step = []
        new_valid_count = []
        new_valid_mask = []
        new_next_obs = []
        interventions = []
        is_before_first_intervention = []
        new_dones = []

        for i, ep in enumerate(replay_data_agent):
            if len(ep.observations) - num_steps_per_chunk >= 0:
                for s in range(len(ep.observations) - num_steps_per_chunk):
                    new_obs.append(ep.observations[s: s + num_steps_per_chunk])
                    new_next_obs.append(ep.next_observations[s: s + num_steps_per_chunk])
                    new_action_behaviors.append(ep.actions_behavior[s: s + num_steps_per_chunk])
                    new_action_novices.append(ep.actions_novice[s: s + num_steps_per_chunk])
                    new_dones.append(ep.dones[s: s + num_steps_per_chunk])
                    new_valid_ep.append(i)
                    new_valid_step.append(s)
                    new_valid_count.append(ep.interventions[s: s + num_steps_per_chunk].sum())
                    new_valid_mask.append(ep.interventions.new_ones(num_steps_per_chunk))

                    intervention = ep.interventions[s: s + num_steps_per_chunk]
                    first_intervention = intervention.squeeze(-1).argmax()
                    interventions.append(intervention)
                    is_before_first_intervention.append(
                        torch.nn.functional.pad(
                            intervention.new_ones(first_intervention + 1), pad=(0, num_steps_per_chunk - first_intervention - 1)
                        )
                    )

            else:
                # Need to pad the data
                new_obs.append(torch.cat([ep.observations, ep.observations.new_zeros(
                    [num_steps_per_chunk - len(ep.observations), *ep.observations.shape[1:]])], dim=0))
                new_next_obs.append(torch.cat([ep.next_observations, ep.next_observations.new_zeros(
                    [num_steps_per_chunk - len(ep.next_observations), *ep.next_observations.shape[1:]])], dim=0))
                new_action_behaviors.append(torch.cat([ep.actions_behavior, ep.actions_behavior.new_zeros(
                    [num_steps_per_chunk - len(ep.actions_behavior), *ep.actions_behavior.shape[1:]])], dim=0))
                new_action_novices.append(torch.cat([ep.actions_novice, ep.actions_novice.new_zeros(
                    [num_steps_per_chunk - len(ep.actions_novice), *ep.actions_novice.shape[1:]])], dim=0))
                new_dones.append(torch.cat([ep.dones, ep.dones.new_zeros(
                    [num_steps_per_chunk - len(ep.dones), *ep.dones.shape[1:]])], dim=0))
                new_valid_ep.append(i)
                new_valid_step.append(0)
                new_valid_count.append(ep.interventions.sum())
                new_valid_mask.append(torch.cat([
                    ep.interventions.new_ones(len(ep.interventions)),
                    ep.interventions.new_zeros(num_steps_per_chunk - len(ep.interventions))
                ]))

                intervention = torch.cat([
                    ep.interventions,
                    ep.interventions.new_zeros(num_steps_per_chunk - len(ep.interventions))
                ])
                first_intervention = intervention.squeeze(-1).argmax()
                interventions.append(intervention)
                is_before_first_intervention.append(
                    torch.nn.functional.pad(
                        intervention.new_ones(first_intervention + 1),
                        pad=(0, num_steps_per_chunk - first_intervention - 1)
                    )
                )

        obs = torch.stack(new_obs)
        actions_behavior = torch.stack(new_action_behaviors)
        actions_novice = torch.stack(new_action_novices)
        next_obs = torch.stack(new_next_obs)
        dones = torch.stack(new_dones)
        interventions = torch.stack(interventions).squeeze(-1)
        is_before_first_intervention = torch.stack(is_before_first_intervention)

        # We also need to prepare RL data:
        rl_obs = []
        rl_next_obs = []
        rl_actions = []
        rl_dones = []
        for i, ep in enumerate(replay_data_agent):
            rl_obs.append(ep.observations)
            rl_next_obs.append(ep.next_observations)
            rl_actions.append(ep.actions_behavior)
            rl_dones.append(ep.dones)
        rl_obs = torch.cat(rl_obs)
        rl_next_obs = torch.cat(rl_next_obs)
        rl_actions = torch.cat(rl_actions)
        rl_dones = torch.cat(rl_dones)

        new_valid_mask = torch.stack(new_valid_mask).bool()
        new_valid_ep = torch.from_numpy(np.array(new_valid_ep)).to(obs.device)
        new_valid_step = torch.from_numpy(np.array(new_valid_step)).to(obs.device)
        new_valid_count = torch.stack(new_valid_count).to(obs.device).int()
        valid_count = new_valid_count
        valid_mask = new_valid_mask

        # Number of chunks to compare
        cpl_bias = self.extra_config["cpl_bias"]

        for step in range(gradient_steps):
            self._n_updates += 1
            alpha = 0.1
            c_ind = None
            num_comparisons = self.extra_config["num_comparisons"]

            cpl_losses = []
            accuracies = []

            assert self.extra_config["use_chunk_adv"]
            assert num_comparisons == -1
            assert self.extra_config["prioritized_buffer"]

            assert (valid_count > 0).any().item(), "No human in the loop data is found."

            human_involved = valid_count > 0
            num_human_involved = human_involved.sum().item()
            stat_recorder["human_ratio"].append(num_human_involved / len(human_involved))

            num_comparisons = num_human_involved

            # Randomly select num_comparisons indices in the human involved data. The indices should in
            # range len(valid_count) not num_human_involved.
            ind = torch.randperm(num_human_involved)
            human_involved_indices = torch.nonzero(human_involved, as_tuple=True)[0]
            no_human_involved_indices = torch.nonzero(~human_involved, as_tuple=True)[0]
            a_ind = human_involved_indices[ind]

            num_c_comparisons = 0
            if len(no_human_involved_indices) > 0:
                # Make the data from agent's exploration equally sized as human involved data.
                c_ind = torch.randint(
                    len(no_human_involved_indices), size=(num_comparisons,)
                ).to(no_human_involved_indices.device)
                num_c_comparisons = num_comparisons

            stat_recorder["num_c_comparisons"].append(num_c_comparisons)

            a_count = valid_count[a_ind]
            a_obs = obs[a_ind]
            a_actions_behavior = actions_behavior[a_ind]
            a_actions_novice = actions_novice[a_ind]

            # Compute advantage for a+, b+, a-, b- trajectory:
            flatten_obs = torch.cat([
                a_obs.flatten(0, 1),
                a_obs.flatten(0, 1),
            ], dim=0)
            flatten_actions = torch.cat([
                a_actions_behavior.flatten(0, 1),
                a_actions_novice.flatten(0, 1),
            ], dim=0)
            flatten_valid_mask = torch.cat([
                valid_mask[a_ind].flatten(),
                valid_mask[a_ind].flatten(),
            ], dim=0)

            # NOTE: to make life easier, we assume q1 is Q net and q2 is value net.
            act = flatten_actions[flatten_valid_mask]
            values = self.reward_model(flatten_obs[flatten_valid_mask], act)
            values = values[0]
            a = values
            full_values = a.new_zeros(flatten_valid_mask.shape[0])
            full_values[flatten_valid_mask] = a.flatten()
            adv_a_pos, adv_a_neg = torch.chunk(full_values, 2)
            adv_a_pos = adv_a_pos.reshape(num_comparisons, num_steps_per_chunk).sum(-1)
            adv_a_neg = adv_a_neg.reshape(num_comparisons, num_steps_per_chunk).sum(-1)

            zeros_label = torch.zeros_like(adv_a_pos)
            # Case 1: a+ > a-
            cpl_loss_1, accuracy_1 = biased_bce_with_logits(adv_a_pos, adv_a_neg, zeros_label, bias=cpl_bias, shuffle=False)
            cpl_losses.append(cpl_loss_1)
            accuracies.append(accuracy_1)

            # Case 3: a+ > b-
            shuffled_indices = torch.randperm(num_comparisons)
            cpl_loss_3, accuracy_3 = biased_bce_with_logits(adv_a_pos, adv_a_neg[shuffled_indices], zeros_label, bias=cpl_bias, shuffle=False)
            cpl_losses.append(cpl_loss_3)
            accuracies.append(accuracy_3)
            stat_recorder["cpl_loss_3"].append(cpl_loss_3.item())
            stat_recorder["cpl_accuracy_3"].append(accuracy_3.item())

            stat_recorder["adv_pos"].append(adv_a_pos.mean().item())
            stat_recorder["adv_neg"].append(adv_a_neg.mean().item())
            cpl_loss = sum(cpl_losses)
            accuracy = sum(accuracies) / len(cpl_losses)
            stat_recorder["cpl_loss_1"].append(cpl_loss_1.item())
            stat_recorder["cpl_accuracy"].append(accuracy.item())
            stat_recorder["cpl_accuracy_1"].append(accuracy_1.item())

            # Optimization step
            self.reward_model.optimizer.zero_grad()
            cpl_loss.backward()
            # Clip grad norm
            self.reward_model.optimizer.step()

            polyak_update(self.reward_model.parameters(), self.reward_model_target.parameters(), self.tau)

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:

                # Compute actor loss
                actor_loss = -self.reward_model.q1_forward(
                    rl_obs, self.actor(rl_obs)
                ).mean()

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

                stat_recorder["actor_loss"].append(actor_loss.item())
                self.actor_update_count += 1

        action_norm = np.linalg.norm(self.policy.predict(rl_obs.cpu(), deterministic=True)[0] - rl_actions.cpu().numpy(), axis=-1).mean()
        gt_norm = (actions_novice - actions_behavior).norm(dim=-1).mean().item()


        self.logger.record("train/pred_action_norm", action_norm)
        self.logger.record("train/gt_action_norm", gt_norm)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        for key, values in stat_recorder.items():
            self.logger.record("train/{}".format(key), np.mean(values))

    def _store_transition(
            self,
            replay_buffer: ReplayBuffer,
            buffer_action: np.ndarray,
            new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
            reward: np.ndarray,
            dones: np.ndarray,
            infos: List[Dict[str, Any]],
    ) -> None:
        # if infos[0]["takeover"] or infos[0]["takeover_start"]:
        #     replay_buffer = self.human_data_buffer
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)

    def save_replay_buffer(
            self, path_human: Union[str, pathlib.Path, io.BufferedIOBase], path_replay: Union[str, pathlib.Path,
            io.BufferedIOBase]
    ) -> None:
        save_to_pkl(path_human, self.human_data_buffer, self.verbose)
        super().save_replay_buffer(path_replay)

    def load_replay_buffer(
            self,
            path_human: Union[str, pathlib.Path, io.BufferedIOBase],
            path_replay: Union[str, pathlib.Path, io.BufferedIOBase],
            truncate_last_traj: bool = True,
    ) -> None:
        """
        Load a replay buffer from a pickle file.

        :param path: Path to the pickled replay buffer.
        :param truncate_last_traj: When using ``HerReplayBuffer`` with online sampling:
            If set to ``True``, we assume that the last trajectory in the replay buffer was finished
            (and truncate it).
            If set to ``False``, we assume that we continue the same trajectory (same episode).
        """
        self.human_data_buffer = load_from_pkl(path_human, self.verbose)
        assert isinstance(
            self.human_data_buffer, ReplayBuffer
        ), "The replay buffer must inherit from ReplayBuffer class"

        # Backward compatibility with SB3 < 2.1.0 replay buffer
        # Keep old behavior: do not handle timeout termination separately
        if not hasattr(self.human_data_buffer, "handle_timeout_termination"):  # pragma: no cover
            self.human_data_buffer.handle_timeout_termination = False
            self.human_data_buffer.timeouts = np.zeros_like(self.replay_buffer.dones)
        super().load_replay_buffer(path_replay, truncate_last_traj)

    def _get_torch_save_params(self):
        ret = super()._get_torch_save_params()
        # print(1)
        return (['policy'], [])

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "run",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
            save_timesteps: int = 2000,
            buffer_save_timesteps: int = 2000,
            save_path_human: Union[str, pathlib.Path, io.BufferedIOBase] = "",
            save_path_replay: Union[str, pathlib.Path, io.BufferedIOBase] = "",
            save_buffer: bool = True,
            load_buffer: bool = False,
            load_path_human: Union[str, pathlib.Path, io.BufferedIOBase] = "",
            load_path_replay: Union[str, pathlib.Path, io.BufferedIOBase] = "",
            warmup: bool = False,
            warmup_steps: int = 5000,
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
        if load_buffer:
            self.load_replay_buffer(load_path_human, load_path_replay)
        callback.on_training_start(locals(), globals())
        if warmup:
            assert load_buffer, "warmup is useful only when load buffer"
            print("Start warmup with steps: " + str(warmup_steps))
            self.train(batch_size=self.batch_size, gradient_steps=warmup_steps)

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
                deterministic=True,  # <<<<< We use deterministic PPO policy here!
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
            if save_buffer and self.num_timesteps > 0 and self.num_timesteps % buffer_save_timesteps == 0:
                buffer_location_human = os.path.join(
                    save_path_human, "human_buffer_" + str(self.num_timesteps) + ".pkl"
                )
                buffer_location_replay = os.path.join(
                    save_path_replay, "replay_buffer_" + str(self.num_timesteps) + ".pkl"
                )
                logger.info("Saving..." + str(buffer_location_human))
                logger.info("Saving..." + str(buffer_location_replay))
                self.save_replay_buffer(buffer_location_human, buffer_location_replay)

        callback.on_training_end()

        return self
