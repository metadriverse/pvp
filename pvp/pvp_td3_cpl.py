import io
import logging
import os
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Union, Optional

import numpy as np
import torch

from pvp.sb3.common.buffers import ReplayBuffer
from pvp.sb3.common.save_util import load_from_pkl, save_to_pkl
from pvp.sb3.common.type_aliases import GymEnv, MaybeCallback
from pvp.sb3.haco.haco_buffer import HACOReplayBuffer
from pvp.sb3.td3.td3 import TD3

logger = logging.getLogger(__name__)


def log_probs_to_advantages(log_probs, alpha, remove_sum=False):
    if remove_sum:
        return (alpha * log_probs)
    return (alpha * log_probs).sum(dim=-1)


def biased_bce_with_logits(adv1, adv2, y, bias=1.0, shuffle=False):
    # Apply the log-sum-exp trick.
    # y = 1 if we prefer x2 to x1
    # We need to implement the numerical stability trick.

    # If shuffle is True, we will shuffle the order of adv1 and adv2. In this case y must be all 0 or 1.
    if shuffle:
        adv1 = adv1[torch.randperm(adv1.shape[0])]
        adv2 = adv2[torch.randperm(adv2.shape[0])]

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


class PVPTD3CPL(TD3):
    actor_update_count = 0

    def __init__(self, use_balance_sample=True, q_value_bound=1., *args, **kwargs):
        """Please find the hyperparameters from original TD3"""
        if "cql_coefficient" in kwargs:
            self.cql_coefficient = kwargs["cql_coefficient"]
            kwargs.pop("cql_coefficient")
        else:
            self.cql_coefficient = 1
        if "replay_buffer_class" not in kwargs:
            kwargs["replay_buffer_class"] = HACOReplayBuffer

        if "intervention_start_stop_td" in kwargs:
            self.intervention_start_stop_td = kwargs["intervention_start_stop_td"]
            kwargs.pop("intervention_start_stop_td")
        else:
            # Default to set it True. We find this can improve the performance and user experience.
            self.intervention_start_stop_td = True

        self.extra_config = {}
        for k in [
            "use_chunk_adv",
            "add_loss_5",
            "prioritized_buffer",
            "mask_same_actions",
            "remove_loss_1",
            "training_deterministic"
        ]:
            if k in kwargs:
                v = kwargs.pop(k)
                assert v in ["True", "False"]
                v = v == "True"
                self.extra_config[k] = v
        for k in [
            "num_comparisons",
            "num_steps_per_chunk",
            "cpl_bias",
            "top_factor"
        ]:
            if k in kwargs:
                v = kwargs.pop(k)
                self.extra_config[k] = v

        self.q_value_bound = q_value_bound
        self.use_balance_sample = use_balance_sample
        super().__init__(*args, **kwargs)

    # def _setup_lr_schedule(self):
    #     from pvp.sb3.common.utils import get_schedule_fn
    #     self.lr_schedule = {k: get_schedule_fn(self.learning_rate[k]) for k in self.learning_rate}

    def _create_aliases(self) -> None:
        pass
        # self.actor = self.policy.actor
        # self.actor_target = self.policy.actor_target
        # self.critic = self.policy.critic
        # self.critic_target = self.policy.critic_target

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
        # else:
        # self.human_data_buffer = self.replay_buffer

    # def _update_learning_rate(self, optimizers: Union[List[th.optim.Optimizer], th.optim.Optimizer]) -> None:
    #     """
    #     Update the optimizers learning rate using the current learning rate schedule
    #     and the current progress remaining (from 1 to 0).
    #
    #     :param optimizers:
    #         An optimizer or a list of optimizers.
    #     """
    #     pass
    # from pvp.sb3.common.utils import update_learning_rate

    # # Log the current learning rate
    # self.logger.record("train/learning_rate", self.lr_schedule(self._current_progress_remaining))
    #
    # if not isinstance(optimizers, list):
    #     optimizers = [optimizers]
    # for optimizer in optimizers:
    #     update_learning_rate(optimizer, self.lr_schedule(self._current_progress_remaining))

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.policy.optimizer])  # , self.critic.optimizer])

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

        interventions = []

        for i, ep in enumerate(replay_data_agent):
            if len(ep.observations) - num_steps_per_chunk >= 0:
                for s in range(len(ep.observations) - num_steps_per_chunk):
                    new_obs.append(ep.observations[s: s + num_steps_per_chunk])
                    new_action_behaviors.append(ep.actions_behavior[s: s + num_steps_per_chunk])
                    new_action_novices.append(ep.actions_novice[s: s + num_steps_per_chunk])

                    new_valid_ep.append(i)
                    new_valid_step.append(s)
                    new_valid_count.append(ep.interventions[s: s + num_steps_per_chunk].sum())
                    new_valid_mask.append(ep.interventions.new_ones(num_steps_per_chunk))

                    interventions.append(ep.interventions[s: s + num_steps_per_chunk])

            else:
                # Need to pad the data
                new_obs.append(torch.cat([ep.observations, ep.observations.new_zeros(
                    [num_steps_per_chunk - len(ep.observations), *ep.observations.shape[1:]])], dim=0))
                new_action_behaviors.append(torch.cat([ep.actions_behavior, ep.actions_behavior.new_zeros(
                    [num_steps_per_chunk - len(ep.actions_behavior), *ep.actions_behavior.shape[1:]])], dim=0))
                new_action_novices.append(torch.cat([ep.actions_novice, ep.actions_novice.new_zeros(
                    [num_steps_per_chunk - len(ep.actions_novice), *ep.actions_novice.shape[1:]])], dim=0))

                new_valid_ep.append(i)
                new_valid_step.append(0)
                new_valid_count.append(ep.interventions.sum())
                new_valid_mask.append(torch.cat([
                    ep.interventions.new_ones(len(ep.interventions)),
                    ep.interventions.new_zeros(num_steps_per_chunk - len(ep.interventions))
                ]))

                interventions.append(torch.cat([
                    ep.interventions,
                    ep.interventions.new_zeros(num_steps_per_chunk - len(ep.interventions))
                ]))

        obs = torch.stack(new_obs)
        actions_behavior = torch.stack(new_action_behaviors)
        actions_novice = torch.stack(new_action_novices)

        interventions = torch.stack(interventions).squeeze(-1)

        # FIXME
        # FIXME
        # FIXME
        # FIXME
        # FIXME
        actions_novice = actions_novice.clamp(-1, 1)

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

            # Pick up top half samples
            # num_left = int(len(valid_count) * self.extra_config["top_factor"])
            # num_left = max(10, num_left)
            # descending_indices = descending_indices[:num_left]

            # num_comparisons = num_human_involved // 2
            num_comparisons = num_human_involved

            # Randomly select num_comparisons indices in the human involved data. The indices should in
            # range len(valid_count) not num_human_involved.
            ind = torch.randperm(num_human_involved)
            human_involved_indices = torch.nonzero(human_involved, as_tuple=True)[0]
            no_human_involved_indices = torch.nonzero(~human_involved, as_tuple=True)[0]
            a_ind = human_involved_indices[ind]
            # b_ind = human_involved_indices[ind[-num_comparisons:]]

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
            a_int = interventions[a_ind]

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

            _, log_probs_tmp, entropy = self.policy.evaluate_actions(
                flatten_obs[flatten_valid_mask], flatten_actions[flatten_valid_mask]
            )
            log_probs = log_probs_tmp.new_zeros(flatten_valid_mask.shape[0])
            log_probs[flatten_valid_mask] = log_probs_tmp

            lp_a_pos, lp_a_neg = torch.chunk(log_probs, 2)

            # Debug code:
            # gt = torch.cat(
            #     [
            #         self.policy.evaluate_actions(a_obs.flatten(0, 1), a_actions_behavior.flatten(0, 1))[1],
            #         self.policy.evaluate_actions(b_obs.flatten(0, 1), b_actions_behavior.flatten(0, 1))[1],
            #         self.policy.evaluate_actions(a_obs.flatten(0, 1), a_actions_novice.flatten(0, 1))[1],
            #         self.policy.evaluate_actions(b_obs.flatten(0, 1), b_actions_novice.flatten(0, 1))[1],
            #      ], dim=0
            # )

            adv_a_pos = log_probs_to_advantages(lp_a_pos.reshape(num_comparisons, num_steps_per_chunk), alpha, remove_sum=False)
            adv_a_neg = log_probs_to_advantages(lp_a_neg.reshape(num_comparisons, num_steps_per_chunk), alpha, remove_sum=False)
            # adv_b_pos = log_probs_to_advantages(lp_b_pos.reshape(num_comparisons, num_steps_per_chunk), alpha)
            # adv_b_neg = log_probs_to_advantages(lp_b_neg.reshape(num_comparisons, num_steps_per_chunk), alpha)

            adv_a_pos2 = log_probs_to_advantages(lp_a_pos.reshape(num_comparisons, num_steps_per_chunk), alpha,
                                                 remove_sum=True)
            adv_a_neg2 = log_probs_to_advantages(lp_a_neg.reshape(num_comparisons, num_steps_per_chunk), alpha,
                                                 remove_sum=True)

            zeros_label = torch.zeros_like(adv_a_pos)
            if not self.extra_config["remove_loss_1"]:
                # Case 1: a+ > a-
                if self.extra_config["mask_same_actions"]:
                    cpl_loss_1, accuracy_1 = biased_bce_with_logits((adv_a_pos2 * a_int).sum(-1), (adv_a_neg2 * a_int).sum(-1), zeros_label, bias=cpl_bias, shuffle=False)
                else:
                    cpl_loss_1, accuracy_1 = biased_bce_with_logits(adv_a_pos, adv_a_neg, zeros_label, bias=cpl_bias, shuffle=False)
                cpl_losses.append(cpl_loss_1)
                accuracies.append(accuracy_1)
                stat_recorder["cpl_loss_1"].append(cpl_loss_1.item())
                stat_recorder["cpl_accuracy_1"].append(accuracy_1.item())

            # Case 3: a+ > b-
            shuffled_indices = torch.randperm(num_comparisons)
            cpl_loss_3, accuracy_3 = biased_bce_with_logits(adv_a_pos, adv_a_neg[shuffled_indices], zeros_label, bias=cpl_bias, shuffle=False)
            cpl_losses.append(cpl_loss_3)
            accuracies.append(accuracy_3)

            # Case 5: a+ > b+ or b+ > a+
            # TODO: FIX THIS
            # label5 = a_count > a_count  # if a_count>b_count, we prefer b as it costs less intervention.
            # label5 = label5.float()
            # label5[b_count == a_count] = 0.5
            # cpl_loss_5, accuracy_5 = biased_bce_with_logits(adv_a_pos, adv_b_pos, label5, bias=cpl_bias, shuffle=False)
            # if self.extra_config["add_loss_5"]:
            #     cpl_losses.append(cpl_loss_5)
            #     accuracies.append(accuracy_5)

            # Compute the c trajectory:
            if c_ind is not None:
                c_obs = obs[c_ind]
                c_actions_behavior = actions_behavior[c_ind]
                c_valid_mask = valid_mask[c_ind].flatten()

                _, log_probs_tmp_c, entropy_c = self.policy.evaluate_actions(
                    c_obs.flatten(0, 1)[c_valid_mask], c_actions_behavior.flatten(0, 1)[c_valid_mask]
                )
                log_probs_c = log_probs_tmp_c.new_zeros(c_valid_mask.shape[0])
                log_probs_c[c_valid_mask] = log_probs_tmp_c
                adv_c = log_probs_to_advantages(
                    log_probs_c.reshape(num_c_comparisons, num_steps_per_chunk), alpha
                )

                # Case 6: c > a- & c > b-
                min_comparison = min(num_c_comparisons, num_comparisons)
                zeros_label_c = zeros_label.new_zeros((min_comparison, ))

                cpl_loss_6, accuracy_6 = biased_bce_with_logits(
                    adv_c, adv_a_neg, zeros_label_c, bias=cpl_bias, shuffle=False
                )
                cpl_losses.append(cpl_loss_6)
                accuracies.append(accuracy_6)
                # cpl_loss_62, accuracy_62 = biased_bce_with_logits(
                #     adv_c[:min_comparison], adv_b_neg[:min_comparison], zeros_label_c, bias=cpl_bias, shuffle=False
                # )
                # cpl_losses.append(cpl_loss_62)
                # accuracies.append(accuracy_62)
                # cpl_loss_6 = (cpl_loss_61 + cpl_loss_62) / 2
                # accuracy_6 = (accuracy_61 + accuracy_62) / 2
                stat_recorder["cpl_loss_6"].append(cpl_loss_6.item())
                stat_recorder["cpl_accuracy_6"].append(accuracy_6.item())

            stat_recorder["adv_pos"].append(adv_a_pos.mean().item())
            stat_recorder["adv_neg"].append(adv_a_neg.mean().item())
            # stat_recorder["int_count_pos"].append(torch.where(a_count > b_count, b_count, a_count).float().mean().item())
            # stat_recorder["int_count_neg"].append(torch.where(a_count < b_count, b_count, a_count).float().mean().item())
            stat_recorder["entropy"].append(entropy.mean().item())

            cpl_loss = sum(cpl_losses)
            accuracy = sum(accuracies) / len(cpl_losses)

            # stat_recorder["cpl_loss_2"].append(cpl_loss_2.item())
            stat_recorder["cpl_loss_3"].append(cpl_loss_3.item())
            # stat_recorder["cpl_loss_4"].append(cpl_loss_4.item())
            # stat_recorder["cpl_loss_5"].append(cpl_loss_5.item())

            stat_recorder["cpl_accuracy"].append(accuracy.item())
            # stat_recorder["cpl_accuracy_2"].append(accuracy_2.item())
            stat_recorder["cpl_accuracy_3"].append(accuracy_3.item())
            # stat_recorder["cpl_accuracy_4"].append(accuracy_4.item())
            # stat_recorder["cpl_accuracy_5"].append(accuracy_5.item())

            # Optimization step
            self.policy.optimizer.zero_grad()
            cpl_loss.backward()
            # Clip grad norm
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            self.policy.optimizer.step()

            self.actor_update_count += 1

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
                deterministic=self.extra_config["training_deterministic"],
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






class PVPRealTD3CPL(TD3):
    actor_update_count = 0

    def __init__(self, use_balance_sample=True, q_value_bound=1., *args, **kwargs):
        """Please find the hyperparameters from original TD3"""
        if "cql_coefficient" in kwargs:
            self.cql_coefficient = kwargs["cql_coefficient"]
            kwargs.pop("cql_coefficient")
        else:
            self.cql_coefficient = 1
        if "replay_buffer_class" not in kwargs:
            kwargs["replay_buffer_class"] = HACOReplayBuffer

        if "intervention_start_stop_td" in kwargs:
            self.intervention_start_stop_td = kwargs["intervention_start_stop_td"]
            kwargs.pop("intervention_start_stop_td")
        else:
            # Default to set it True. We find this can improve the performance and user experience.
            self.intervention_start_stop_td = True

        self.extra_config = {}
        for k in [
            "use_chunk_adv",
            "add_loss_5",
            "prioritized_buffer"
        ]:
            if k in kwargs:
                v = kwargs.pop(k)
                assert v in ["True", "False"]
                v = v == "True"
                self.extra_config[k] = v
        for k in [
            "num_comparisons",
            "num_steps_per_chunk",
            "cpl_bias",
            "top_factor"
        ]:
            if k in kwargs:
                v = kwargs.pop(k)
                self.extra_config[k] = v

        self.q_value_bound = q_value_bound
        self.use_balance_sample = use_balance_sample
        super().__init__(*args, **kwargs)

    # def _setup_lr_schedule(self):
    #     from pvp.sb3.common.utils import get_schedule_fn
    #     self.lr_schedule = {k: get_schedule_fn(self.learning_rate[k]) for k in self.learning_rate}

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

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
        # else:
        # self.human_data_buffer = self.replay_buffer

    # def _update_learning_rate(self, optimizers: Union[List[th.optim.Optimizer], th.optim.Optimizer]) -> None:
    #     """
    #     Update the optimizers learning rate using the current learning rate schedule
    #     and the current progress remaining (from 1 to 0).
    #
    #     :param optimizers:
    #         An optimizer or a list of optimizers.
    #     """
    #     pass
    # from pvp.sb3.common.utils import update_learning_rate

    # # Log the current learning rate
    # self.logger.record("train/learning_rate", self.lr_schedule(self._current_progress_remaining))
    #
    # if not isinstance(optimizers, list):
    #     optimizers = [optimizers]
    # for optimizer in optimizers:
    #     update_learning_rate(optimizer, self.lr_schedule(self._current_progress_remaining))

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

        for i, ep in enumerate(replay_data_agent):
            if len(ep.observations) - num_steps_per_chunk >= 0:
                for s in range(len(ep.observations) - num_steps_per_chunk):
                    new_obs.append(ep.observations[s: s + num_steps_per_chunk])
                    new_action_behaviors.append(ep.actions_behavior[s: s + num_steps_per_chunk])
                    new_action_novices.append(ep.actions_novice[s: s + num_steps_per_chunk])

                    new_valid_ep.append(i)
                    new_valid_step.append(s)
                    new_valid_count.append(ep.interventions[s: s + num_steps_per_chunk].sum())
                    new_valid_mask.append(ep.interventions.new_ones(num_steps_per_chunk))

            else:
                # Need to pad the data
                new_obs.append(torch.cat([ep.observations, ep.observations.new_zeros(
                    [num_steps_per_chunk - len(ep.observations), *ep.observations.shape[1:]])], dim=0))
                new_action_behaviors.append(torch.cat([ep.actions_behavior, ep.actions_behavior.new_zeros(
                    [num_steps_per_chunk - len(ep.actions_behavior), *ep.actions_behavior.shape[1:]])], dim=0))
                new_action_novices.append(torch.cat([ep.actions_novice, ep.actions_novice.new_zeros(
                    [num_steps_per_chunk - len(ep.actions_novice), *ep.actions_novice.shape[1:]])], dim=0))

                new_valid_ep.append(i)
                new_valid_step.append(0)
                new_valid_count.append(ep.interventions.sum())
                new_valid_mask.append(torch.cat([
                    ep.interventions.new_ones(len(ep.interventions)),
                    ep.interventions.new_zeros(num_steps_per_chunk - len(ep.interventions))
                ]))

        obs = torch.stack(new_obs)
        actions_behavior = torch.stack(new_action_behaviors)
        actions_novice = torch.stack(new_action_novices)

        # FIXME
        # FIXME
        # FIXME
        # FIXME
        # FIXME
        actions_novice = actions_novice.clamp(-1, 1)

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

            # Pick up top half samples
            # num_left = int(len(valid_count) * self.extra_config["top_factor"])
            # num_left = max(10, num_left)
            # descending_indices = descending_indices[:num_left]

            # num_comparisons = num_human_involved // 2
            num_comparisons = num_human_involved

            # Randomly select num_comparisons indices in the human involved data. The indices should in
            # range len(valid_count) not num_human_involved.
            ind = torch.randperm(num_human_involved)
            human_involved_indices = torch.nonzero(human_involved, as_tuple=True)[0]
            no_human_involved_indices = torch.nonzero(~human_involved, as_tuple=True)[0]
            a_ind = human_involved_indices[ind]
            # b_ind = human_involved_indices[ind[-num_comparisons:]]

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

            # b_count = valid_count[b_ind]
            # b_obs = obs[b_ind]
            # b_actions_behavior = actions_behavior[b_ind]
            # b_actions_novice = actions_novice[b_ind]

            # Compute advantage for a+, b+, a-, b- trajectory:
            flatten_obs = torch.cat([
                a_obs.flatten(0, 1),
                # b_obs.flatten(0, 1),
                a_obs.flatten(0, 1),
                # b_obs.flatten(0, 1)
            ], dim=0)
            flatten_actions = torch.cat([
                a_actions_behavior.flatten(0, 1),
                # b_actions_behavior.flatten(0, 1),
                a_actions_novice.flatten(0, 1),
                # b_actions_novice.flatten(0, 1)
            ], dim=0)
            flatten_valid_mask = torch.cat([
                valid_mask[a_ind].flatten(),
                # valid_mask[b_ind].flatten(),
                valid_mask[a_ind].flatten(),
                # valid_mask[b_ind].flatten()
            ], dim=0)

            # flatten_actions = flatten_actions.clamp(-1, 1)

            # NOTE: to make life easier, we assume q1 is Q net and q2 is value net.
            act = flatten_actions[flatten_valid_mask]
            q = self.critic.q1_forward(flatten_obs[flatten_valid_mask], act)
            v = self.critic.q2_forward(flatten_obs[flatten_valid_mask], torch.zeros_like(act))
            a = q - v
            # values = torch.cat(values, dim=1)
            # values, _ = torch.min(values, dim=1, keepdim=True)
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

            # Case 5: a+ > b+ or b+ > a+
            # TODO: FIX THIS
            # label5 = a_count > a_count  # if a_count>b_count, we prefer b as it costs less intervention.
            # label5 = label5.float()
            # label5[b_count == a_count] = 0.5
            # cpl_loss_5, accuracy_5 = biased_bce_with_logits(adv_a_pos, adv_b_pos, label5, bias=cpl_bias, shuffle=False)
            # if self.extra_config["add_loss_5"]:
            #     cpl_losses.append(cpl_loss_5)
            #     accuracies.append(accuracy_5)

            # Compute the c trajectory:
            if c_ind is not None:
                c_obs = obs[c_ind]
                c_actions_behavior = actions_behavior[c_ind]
                c_valid_mask = valid_mask[c_ind].flatten()

                _, log_probs_tmp_c, entropy_c = self.policy.evaluate_actions(
                    c_obs.flatten(0, 1)[c_valid_mask], c_actions_behavior.flatten(0, 1)[c_valid_mask]
                )
                log_probs_c = log_probs_tmp_c.new_zeros(c_valid_mask.shape[0])
                log_probs_c[c_valid_mask] = log_probs_tmp_c
                adv_c = log_probs_to_advantages(
                    log_probs_c.reshape(num_c_comparisons, num_steps_per_chunk), alpha
                )

                # Case 6: c > a- & c > b-
                min_comparison = min(num_c_comparisons, num_comparisons)
                zeros_label_c = zeros_label.new_zeros((min_comparison, ))
                cpl_loss_6, accuracy_6 = biased_bce_with_logits(
                    adv_c, adv_a_neg, zeros_label_c, bias=cpl_bias, shuffle=False
                )
                cpl_losses.append(cpl_loss_6)
                accuracies.append(accuracy_6)
                stat_recorder["cpl_loss_6"].append(cpl_loss_6.item())
                stat_recorder["cpl_accuracy_6"].append(accuracy_6.item())

            stat_recorder["adv_pos"].append(adv_a_pos.mean().item())
            stat_recorder["adv_neg"].append(adv_a_neg.mean().item())
            cpl_loss = sum(cpl_losses)
            accuracy = sum(accuracies) / len(cpl_losses)
            stat_recorder["cpl_loss_1"].append(cpl_loss_1.item())
            stat_recorder["cpl_loss_3"].append(cpl_loss_3.item())
            stat_recorder["cpl_accuracy"].append(accuracy.item())
            stat_recorder["cpl_accuracy_1"].append(accuracy_1.item())
            stat_recorder["cpl_accuracy_3"].append(accuracy_3.item())

            # Optimization step
            self.critic.optimizer.zero_grad()
            cpl_loss.backward()
            # Clip grad norm
            self.critic.optimizer.step()



            # Policy loss:

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:

                # Compute actor loss
                actor_loss = -self.critic.q1_forward(
                    obs[valid_mask], self.actor(obs[valid_mask])
                ).mean()

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                # polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

                stat_recorder["actor_loss"].append(actor_loss.item())
                self.actor_update_count += 1

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
