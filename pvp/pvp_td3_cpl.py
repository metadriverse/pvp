import io
import logging
import os
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Union, Optional
import copy
import numpy as np
import torch
from pvp.sb3.common.utils import polyak_update
from pvp.sb3.common.buffers import ReplayBuffer
from pvp.sb3.common.save_util import load_from_pkl, save_to_pkl
from pvp.sb3.common.type_aliases import GymEnv, MaybeCallback
from pvp.sb3.haco.haco_buffer import HACOReplayBuffer
from pvp.sb3.td3.td3 import TD3
# from pvp.sb3.dqn.policies import DQNPolicy,QNetwork

from pvp.sb3.ppo.policies import ActorCriticPolicy

class PVPTD3CPLPolicy(ActorCriticPolicy):

    def __init__(self, obss, acts, *args, **kwargs):
        for k in ["fixed_log_std", "log_std_init"]:
            if k in kwargs:
                kwargs.pop(k)

        self.num_bins = 13
        total_num_bins = self.num_bins ** acts.shape[0]

        self.raw_action_space = acts

        from gym.spaces import Discrete
        super().__init__(obss, Discrete(total_num_bins), *args, **kwargs)

        self.num_axes = len(acts.low)

        # Compute the bin sizes for each axis and prepare the flattened lookup table
        self.lookup_table = torch.zeros(self.num_bins ** self.num_axes, self.num_axes)
        ranges = [torch.linspace(acts.low[axis], acts.high[axis], self.num_bins)
                  for axis in range(self.num_axes)]

        grid = torch.meshgrid(*ranges, indexing='ij')
        # Flatten the grid and store it in the lookup table
        for i in range(self.num_axes):
            self.lookup_table[:, i] = grid[i].flatten()

    def evaluate_actions(self, obs, act):
        dact = torch.cdist(act, self.lookup_table).argmin(-1)
        out = super().evaluate_actions(obs, dact)
        return out

    def raw_predict(self, obs, deterministic=False):
        discrete_action = super()._predict(obs, deterministic)
        return discrete_action

    def scale_action(self, action):
        low, high = self.raw_action_space.low, self.raw_action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        low, high = self.raw_action_space.low, self.raw_action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def _predict(self, obs, state=None, episode_start=None, deterministic=False):
        if self.lookup_table.device != obs.device:
            self.lookup_table = self.lookup_table.to(self.device)

        discrete_action = self.raw_predict(obs, deterministic=deterministic)

        # Use advanced indexing to map discrete actions to continuous actions
        continuous_action = self.lookup_table[discrete_action]
        return continuous_action


logger = logging.getLogger(__name__)

def unwrap(tensor, mask):
    new = tensor.new_zeros(mask.shape)
    new[mask] = tensor
    return new

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
            "add_loss_5_inverse",
            "prioritized_buffer",
            "mask_same_actions",
            "remove_loss_1",
            "remove_loss_3",
            "remove_loss_6",
            "training_deterministic",
            "use_target_policy",
            "use_target_policy_only_overwrite_takeover",
            "add_bc_loss",
            "add_bc_loss_only_interventions"
        ]:
            if k in kwargs:
                v = kwargs.pop(k)
                assert v in ["True", "False", True, False]
                if isinstance(v, str):
                    v = v == "True"
                self.extra_config[k] = v
        for k in [
            "num_comparisons",
            "num_steps_per_chunk",
            "cpl_bias",
            "top_factor",
            "last_ratio",
            "max_comparisons",
            "hard_reset",
            "bc_loss_weight"
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
        self.policy_target = copy.deepcopy(self.policy)
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
        is_before_first_intervention = []

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

        interventions = torch.stack(interventions).squeeze(-1)
        is_before_first_intervention = torch.stack(is_before_first_intervention)

        # actions_novice_noclamp = actions_novice
        # actions_novice = actions_novice.clamp(-1, 1)

        new_valid_mask = torch.stack(new_valid_mask).bool()
        new_valid_ep = torch.from_numpy(np.array(new_valid_ep)).to(obs.device)
        new_valid_step = torch.from_numpy(np.array(new_valid_step)).to(obs.device)
        new_valid_count = torch.stack(new_valid_count).to(obs.device).int()
        valid_count = new_valid_count
        valid_mask = new_valid_mask

        if self.extra_config["last_ratio"] > 0:
            num_samples = int(len(valid_count) * self.extra_config["last_ratio"])
            START_SAMPLES = 1024
            num_samples = max(START_SAMPLES, num_samples)
            if len(valid_count) >= START_SAMPLES:
                print("Sample from the last part of the data. Samples: ", num_samples)
                # valid_count, indices = valid_count.topk(num_samples, largest=False)
                valid_mask = valid_mask[-num_samples:].clone()
                obs = obs[-num_samples:].clone()
                actions_behavior = actions_behavior[-num_samples:].clone()
                actions_novice = actions_novice[-num_samples:].clone()
                interventions = interventions[-num_samples:].clone()
                valid_count = valid_count[-num_samples:].clone()

        # Number of chunks to compare
        cpl_bias = self.extra_config["cpl_bias"]

        # TODO REMOVE
        # first_chunk = valid_count.nonzero()[0].item()
        # first_step = interventions[first_chunk].nonzero()[0].item()
        # print("Action behavior: ", actions_behavior[first_chunk, first_step])
        # print("Action novice: ", actions_novice[first_chunk, first_step])

        rl_obs = []
        rl_next_obs = []
        rl_actions = []
        rl_actions_novice = []
        rl_dones = []
        rl_interventions = []
        for i, ep in enumerate(replay_data_agent):
            rl_obs.append(ep.observations)
            rl_next_obs.append(ep.next_observations)
            rl_actions.append(ep.actions_behavior)
            rl_actions_novice.append(ep.actions_novice)
            rl_dones.append(ep.dones)
            rl_interventions.append(ep.interventions)
        rl_obs = torch.cat(rl_obs)
        rl_next_obs = torch.cat(rl_next_obs)
        rl_actions = torch.cat(rl_actions)
        rl_dones = torch.cat(rl_dones)
        rl_interventions = torch.cat(rl_interventions)
        rl_interventions = rl_interventions.flatten().bool()
        rl_actions_novice = torch.cat(rl_actions_novice)

        # if self.extra_config["use_chunk_adv"]:
        # Reorganize data with chunks
        # Now obs.shape = (#batches, #steps, #features)
        # We need to make it to be: obs.shape = (#batches, #steps-chunk_size, chunk_size, #features)
        full_obs = []
        full_action_behaviors = []
        full_action_novices = []
        full_interventions = []
        full_num_steps_per_chunk = 1000
        for i, ep in enumerate(replay_data_agent):
            # Need to pad the data
            if len(ep.observations) > full_num_steps_per_chunk:
                full_obs.append(ep.observations[:full_num_steps_per_chunk])
                full_action_behaviors.append(ep.actions_behavior[:full_num_steps_per_chunk])
                full_action_novices.append(ep.actions_novice[:full_num_steps_per_chunk])
                full_interventions.append(ep.interventions.flatten()[:full_num_steps_per_chunk])
            else:
                full_obs.append(torch.cat([ep.observations, ep.observations.new_zeros(
                    [full_num_steps_per_chunk - len(ep.observations), *ep.observations.shape[1:]])], dim=0))
                full_action_behaviors.append(torch.cat([ep.actions_behavior, ep.actions_behavior.new_zeros(
                    [full_num_steps_per_chunk - len(ep.actions_behavior), *ep.actions_behavior.shape[1:]])], dim=0))
                full_action_novices.append(torch.cat([ep.actions_novice, ep.actions_novice.new_zeros(
                    [full_num_steps_per_chunk - len(ep.actions_novice), *ep.actions_novice.shape[1:]])], dim=0))
                full_intervention = torch.cat([
                    ep.interventions.flatten(),
                    ep.interventions.new_zeros(full_num_steps_per_chunk - len(ep.interventions))
                ])
                full_interventions.append(full_intervention)
        full_obs = torch.stack(full_obs, dim=0)
        full_action_behaviors = torch.stack(full_action_behaviors, dim=0)
        full_action_novices = torch.stack(full_action_novices, dim=0)
        full_interventions = torch.stack(full_interventions).bool()


        if self.extra_config["hard_reset"] > 0:
            if (self.since_last_reset - self.extra_config["hard_reset"]) >= 0:
                self.policy.reset_parameters()
                print("Hard reset the policy. Since last step: ", self.since_last_reset)
                self.since_last_reset = 0

                # TODO: Policy target??
                # self.policy_target.reset()



        for step in range(gradient_steps):

            # TODO: REMOVE
            # if step % 100 == 0 or step == gradient_steps - 1:
            #     print("STEP", step, self.policy.predict(obs[first_chunk, first_step].cpu(), deterministic=True)[0])

            self._n_updates += 1
            alpha = 0.1
            c_ind = None
            num_comparisons = self.extra_config["num_comparisons"]

            cpl_losses = []
            accuracies = []

            assert self.extra_config["use_chunk_adv"]
            assert self.extra_config["prioritized_buffer"]

            assert (valid_count > 0).any().item(), "No human in the loop data is found."

            human_involved = valid_count > 0
            num_human_involved = human_involved.sum().item()
            stat_recorder["human_ratio"].append(num_human_involved / len(human_involved))

            # Pick up top half samples
            # num_left = int(len(valid_count) * self.extra_config["top_factor"])
            # num_left = max(10, num_left)
            # descending_indices = descending_indices[:num_left]

            # Hard limit the number of comparisons to avoid GPU OOM
            if num_comparisons < 0:
                num_comparisons = min(num_human_involved, self.extra_config["max_comparisons"])
            else:
                num_comparisons = min(num_human_involved, num_comparisons)

            # Randomly select num_comparisons indices in the human involved data. The indices should in
            # range len(valid_count) not num_human_involved.
            ind = torch.randperm(num_human_involved)
            ind = ind[:num_comparisons]

            human_involved_indices = torch.nonzero(human_involved, as_tuple=True)[0]
            no_human_involved_indices = torch.nonzero(~human_involved, as_tuple=True)[0]
            a_ind = human_involved_indices[ind]
            # b_ind = human_involved_indices[ind[-num_comparisons:]]

            a_count = valid_count[a_ind]
            a_obs = obs[a_ind]
            a_actions_behavior = actions_behavior[a_ind]
            a_actions_novice = actions_novice[a_ind]
            a_int = interventions[a_ind]

            # Compute advantage for a+, b+, a-, b- trajectory:


            if self.extra_config["use_target_policy"]:
                m = valid_mask[a_ind].flatten()
                _, log_probs_tmp1, entropy1 = self.policy.evaluate_actions(
                    a_obs.flatten(0, 1)[m], a_actions_behavior.flatten(0, 1)[m]
                )
                lp_a_pos = log_probs_tmp1.new_zeros(m.shape[0])
                lp_a_pos[m] = log_probs_tmp1

                with torch.no_grad():
                    a_actions_novice_target = self.policy_target._predict(a_obs.flatten(0, 1)[m], deterministic=False)

                if self.extra_config["use_target_policy_only_overwrite_takeover"]:
                    int_mask = a_int.flatten(0, 1)
                    a_actions_novice = torch.where(
                        (int_mask == 1)[:, None], a_actions_novice_target, a_actions_novice.flatten(0, 1)[m]
                    )

                else:
                    a_actions_novice = a_actions_novice_target

                _, log_probs_tmp2, entropy2 = self.policy.evaluate_actions(
                    a_obs.flatten(0, 1)[m], a_actions_novice
                )
                lp_a_neg = log_probs_tmp2.new_zeros(m.shape[0])
                lp_a_neg[m] = log_probs_tmp2

                entropy = entropy1

            else:
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

                stat_recorder["log_probs"].append(log_probs_tmp.mean().item())

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

            # TODO: Remove debug code:
            adv_a_pos2 = log_probs_to_advantages(lp_a_pos.reshape(num_comparisons, num_steps_per_chunk), alpha,
                                                 remove_sum=True)
            adv_a_neg2 = log_probs_to_advantages(lp_a_neg.reshape(num_comparisons, num_steps_per_chunk), alpha,
                                                 remove_sum=True)
            nppos = adv_a_pos2.cpu().detach().numpy()
            npneg = adv_a_neg2.cpu().detach().numpy()
            inte = interventions[a_ind].cpu().detach().numpy()
            nppos2 = nppos * inte
            npneg2 = npneg * inte


            if self.extra_config["add_bc_loss"]:
                # assert self.extra_config["remove_loss_1"]
                # assert self.extra_config["remove_loss_3"]
                # assert self.extra_config["remove_loss_6"]
                # assert not self.extra_config["add_loss_5"]

                if self.extra_config["add_bc_loss_only_interventions"]:
                    lp = self.policy.evaluate_actions(rl_obs[rl_interventions], rl_actions[rl_interventions])[1]
                else:
                    lp = self.policy.evaluate_actions(rl_obs, rl_actions)[1]
                bc_loss = -lp.mean()

                cpl_losses.append(bc_loss)



            zeros_label = torch.zeros_like(adv_a_pos)
            if not self.extra_config["remove_loss_1"]:
                # Case 1: a+ > a-
                # if self.extra_config["mask_same_actions"]:
                #
                #     # Create a mask so that after the first step where intervention happens the mask is all zeros.
                #     before_int = is_before_first_intervention[a_ind]
                #     cpl_loss_1, accuracy_1 = biased_bce_with_logits((adv_a_pos2 * before_int).sum(-1), (adv_a_neg2 * before_int).sum(-1), zeros_label, bias=cpl_bias, shuffle=False)
                # else:
                #     cpl_loss_1, accuracy_1 = biased_bce_with_logits(adv_a_pos, adv_a_neg, zeros_label, bias=cpl_bias, shuffle=False)

                if self.extra_config["num_comparisons"] < 0:
                    loss1_pos_lp = self.policy.evaluate_actions(rl_obs[rl_interventions], rl_actions[rl_interventions])[1]
                    loss1_neg_lp = self.policy.evaluate_actions(rl_obs[rl_interventions], rl_actions_novice[rl_interventions])[1]
                else:
                    rl_ind = torch.randint(
                        len(rl_obs[rl_interventions]), size=(self.extra_config["num_comparisons"],)
                    ).to(no_human_involved_indices.device)
                    loss1_pos_lp = self.policy.evaluate_actions(rl_obs[rl_interventions][rl_ind], rl_actions[rl_interventions][rl_ind])[1]
                    loss1_neg_lp = self.policy.evaluate_actions(rl_obs[rl_interventions][rl_ind], rl_actions_novice[rl_interventions][rl_ind])[1]

                loss1_adv_pos = loss1_pos_lp * alpha
                loss1_adv_neg = loss1_neg_lp * alpha
                loss1_cpl_bias = 1.0


                # loss1_lp_pos = unwrap(
                #     self.policy.evaluate_actions(full_obs[full_interventions], full_action_behaviors[full_interventions])[1],
                #     full_interventions
                # )
                # loss1_adv_pos = log_probs_to_advantages(loss1_lp_pos, alpha)
                #
                # loss1_lp_neg = unwrap(
                #     self.policy.evaluate_actions(full_obs[full_interventions], full_action_novices[full_interventions])[1],
                #     full_interventions
                # )
                # loss1_adv_neg = log_probs_to_advantages(loss1_lp_neg, alpha)

                cpl_loss_1, accuracy_1 = biased_bce_with_logits(
                    loss1_adv_pos, loss1_adv_neg, torch.zeros_like(loss1_adv_neg), bias=loss1_cpl_bias,
                )

                cpl_losses.append(cpl_loss_1)
                accuracies.append(accuracy_1)
                stat_recorder["cpl_loss_1"].append(cpl_loss_1.item())
                stat_recorder["cpl_accuracy_1"].append(accuracy_1.item())

                bc_loss = -loss1_pos_lp.mean()
                stat_recorder["bc_loss"].append(bc_loss.item())
                if self.extra_config["bc_loss_weight"] > 0:
                    cpl_losses.append(bc_loss * self.extra_config["bc_loss_weight"])

            # Case 3: a+ > b-
            if not self.extra_config["remove_loss_3"]:
                shuffled_indices = torch.randperm(num_comparisons)
                cpl_loss_3, accuracy_3 = biased_bce_with_logits(adv_a_pos, adv_a_neg[shuffled_indices], zeros_label, bias=cpl_bias, shuffle=False)
                cpl_losses.append(cpl_loss_3)
                accuracies.append(accuracy_3)
                stat_recorder["cpl_loss_3"].append(cpl_loss_3.item())
                stat_recorder["cpl_accuracy_3"].append(accuracy_3.item())

            # Case 5: a+ > b+ or b+ > a+
            if self.extra_config["add_loss_5"]:
                shuffled_indices5 = torch.randperm(num_comparisons)

                b_count = valid_count[ind][shuffled_indices5]
                a_count = valid_count[ind]

                if self.extra_config["add_loss_5_inverse"]:
                    label5 = (a_count > b_count).float()
                else:
                    label5 = (a_count < b_count).float()
                label5[a_count == b_count] = 0.5

                cpl_loss_5, accuracy_5 = biased_bce_with_logits(
                    adv_a_pos, adv_a_pos[shuffled_indices5], label5, bias=cpl_bias, shuffle=False)

                cpl_losses.append(cpl_loss_5)
                accuracies.append(accuracy_5)

            # Compute the c trajectory:

            num_c_comparisons = 0
            if len(no_human_involved_indices) > 0 and (not self.extra_config["remove_loss_6"]):
                # Make the data from agent's exploration equally sized as human involved data.
                c_ind = torch.randint(
                    len(no_human_involved_indices), size=(num_comparisons,)
                ).to(no_human_involved_indices.device)
                num_c_comparisons = num_comparisons

                # This is very important!! We need to map the indices back to the original indices.
                c_ind = no_human_involved_indices[c_ind]

                c_obs = obs[c_ind]
                c_actions_behavior = actions_behavior[c_ind]
                c_actions_novice = actions_novice[c_ind]
                c_valid_mask = valid_mask[c_ind].flatten()

                # TODO: Remove a quick test
                assert (c_actions_novice == c_actions_novice).all()

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

            stat_recorder["num_comparisons"].append(num_comparisons)
            stat_recorder["num_c_comparisons"].append(num_c_comparisons)
            stat_recorder["adv_pos"].append(adv_a_pos.mean().item())
            stat_recorder["adv_neg"].append(adv_a_neg.mean().item())
            # stat_recorder["int_count_pos"].append(torch.where(a_count > b_count, b_count, a_count).float().mean().item())
            # stat_recorder["int_count_neg"].append(torch.where(a_count < b_count, b_count, a_count).float().mean().item())
            stat_recorder["entropy"].append(entropy.mean().item())

            cpl_loss = sum(cpl_losses)
            accuracy = sum(accuracies) / len(cpl_losses) if accuracies else None

            # stat_recorder["cpl_loss_2"].append(cpl_loss_2.item())
            # stat_recorder["cpl_loss_4"].append(cpl_loss_4.item())
            # stat_recorder["cpl_loss_5"].append(cpl_loss_5.item())

            stat_recorder["cpl_accuracy"].append(accuracy.item() if accuracy else float("nan"))
            # stat_recorder["cpl_accuracy_2"].append(accuracy_2.item())
            # stat_recorder["cpl_accuracy_4"].append(accuracy_4.item())
            # stat_recorder["cpl_accuracy_5"].append(accuracy_5.item())

            # Optimization step
            self.policy.optimizer.zero_grad()
            cpl_loss.backward()
            # Clip grad norm
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            self.policy.optimizer.step()

            polyak_update(self.policy.parameters(), self.policy_target.parameters(), self.tau)

            self.actor_update_count += 1

        action_norm = np.linalg.norm(
            self.policy.predict(obs.cpu().flatten(0, 1), deterministic=True)[0]
            - actions_behavior.flatten(0, 1).cpu().numpy(),
            axis=-1).mean()
        gt_norm = (actions_novice - actions_behavior).norm(dim=-1).mean().item()
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/pred_action_norm", action_norm)
        self.logger.record("train/gt_action_norm", gt_norm)
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
            eval_deterministic=True,
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
            eval_deterministic
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
