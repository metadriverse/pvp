import io
import logging
import os
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Union, Optional

import numpy as np
import torch as th
from torch.nn import functional as F

from pvp.sb3.common.buffers import ReplayBuffer
from pvp.sb3.common.save_util import load_from_pkl, save_to_pkl
from pvp.sb3.common.type_aliases import GymEnv, MaybeCallback
from pvp.sb3.common.utils import polyak_update
from pvp.sb3.haco.haco_buffer import HACOReplayBuffer, concat_samples
from pvp.sb3.td3.td3 import TD3
import torch

logger = logging.getLogger(__name__)


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


class PVPTD3CPL(TD3):
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
                "chunk_steps",
        ]:
            if k in kwargs:
                v = kwargs.pop(k)
                assert v in ["True", "False"]
                v = v == "True"
                self.extra_config[k] = v
        for k in [
                "num_comparisons",
                "chunk_steps",
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
        # TODO: Not update
        # self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        stat_recorder = defaultdict(list)

        # Sample replay buffer
        # replay_data_human = None
        replay_data_agent = None
        # if self.replay_buffer.pos > 0 and self.human_data_buffer.pos > 0:
        #     replay_data_agent = self.replay_buffer.sample(batch_size, num_steps=num_steps_per_chunk, env=self._vec_normalize_env)
        #     replay_data_human = self.human_data_buffer.sample(batch_size, num_steps=num_steps_per_chunk, env=self._vec_normalize_env)
        # elif self.human_data_buffer.pos > 0:
        #     replay_data_human = self.human_data_buffer.sample(batch_size, num_steps=num_steps_per_chunk, env=self._vec_normalize_env)
        if self.replay_buffer.pos > 0:
            replay_data_agent = self.replay_buffer.sample(0, env=self._vec_normalize_env)
        else:
            replay_data_agent = None

        if replay_data_agent is None:
            return

        # print(111)
        s_list = [len(ep.observations) for ep in replay_data_agent]
        max_s = max(s_list)
        bs = len(s_list)


        # TODO: CONFIG!
        num_steps_per_chunk = 64

        obs = replay_data_agent[0].observations.new_zeros([bs, max_s, replay_data_agent[0].observations.shape[-1]])
        mask = replay_data_agent[0].observations.new_zeros([bs, max_s])
        actions_behavior = replay_data_agent[0].actions_behavior.new_zeros(
            [bs, max_s, replay_data_agent[0].actions_behavior.shape[-1]]
        )
        actions_novice = replay_data_agent[0].actions_novice.new_zeros(
            [bs, max_s, replay_data_agent[0].actions_novice.shape[-1]]
        )
        interventions = replay_data_agent[0].interventions.new_zeros([bs, max_s])
        # intervention_count = replay_data_agent[0].interventions.new_zeros([bs, max_s - num_steps_per_chunk])

        valid_ep = []
        valid_step = []
        valid_count = []

        for i, ep in enumerate(replay_data_agent):
            obs[i, :len(ep.observations)] = ep.observations
            mask[i, :len(ep.observations)] = 1
            actions_behavior[i, :len(ep.actions_behavior)] = ep.actions_behavior
            actions_novice[i, :len(ep.actions_novice)] = ep.actions_novice
            interventions[i, :len(ep.interventions)] = ep.interventions.reshape(-1)
            epint = ep.interventions.reshape(-1)
            if len(ep.observations) >= num_steps_per_chunk:
                valid_ep.extend([i for s in range(len(ep.observations) - num_steps_per_chunk)])
                valid_step.extend(list(range(len(ep.observations) - num_steps_per_chunk)))
                valid_count.extend([
                    epint[s: s+num_steps_per_chunk].sum() for s in range(len(ep.observations) - num_steps_per_chunk)
                ])
            else:
                valid_ep.append(i)
                valid_step.append(0)
                valid_count.append(epint.sum())
        interventions = interventions.bool()


        valid_ep = torch.from_numpy(np.array(valid_ep)).to(interventions.device)
        valid_step = torch.from_numpy(np.array(valid_step)).to(interventions.device)
        valid_count = torch.stack(valid_count).to(interventions.device).int()

        if self.extra_config["use_chunk_adv"]:
            # Reorganize data with chunks
            # Now obs.shape = (#batches, #steps, #features)
            # We need to make it to be: obs.shape = (#batches, #steps-chunk_size, chunk_size, #features)
            new_obs = []
            new_action_behaviors = []
            new_action_novices = []
            for i, ep in enumerate(replay_data_agent):
                if len(ep.observations) - num_steps_per_chunk >= 0:
                    for s in range(len(ep.observations) - num_steps_per_chunk):
                        new_obs.append(ep.observations[s: s + num_steps_per_chunk])
                        new_action_behaviors.append(ep.actions_behavior[s: s + num_steps_per_chunk])
                        new_action_novices.append(ep.actions_novice[s: s + num_steps_per_chunk])
                else:
                    # Need to pad the data
                    new_obs.append(torch.cat([ep.observations, ep.observations.new_zeros([num_steps_per_chunk - len(ep.observations), *ep.observations.shape[1:]])], dim=0))
                    new_action_behaviors.append(torch.cat([ep.actions_behavior, ep.actions_behavior.new_zeros([num_steps_per_chunk - len(ep.actions_behavior), *ep.actions_behavior.shape[1:]])], dim=0))
                    new_action_novices.append(torch.cat([ep.actions_novice, ep.actions_novice.new_zeros([num_steps_per_chunk - len(ep.actions_novice), *ep.actions_novice.shape[1:]])], dim=0))

            # print(2222)

            obs = torch.stack(new_obs)
            actions_behavior = torch.stack(new_action_behaviors)
            actions_novice = torch.stack(new_action_novices)



        # Number of chunks to compare
        num_comparisons = self.extra_config["num_comparisons"]

        for step in range(gradient_steps):
            self._n_updates += 1

            # if replay_data_human is not None and replay_data_agent is not None:
            #     replay_data = concat_samples(replay_data_agent, replay_data_human)
            # else:
            #     replay_data = replay_data_agent if replay_data_agent is not None else replay_data_human

            # ========== Compute our CPL loss here (only train the advantage function) ==========
            # The policy will be trained to maximize the advantage function.
            accuracy = cpl_loss = bc_loss = None

            alpha = 0.1

            if self.extra_config["use_chunk_adv"]:

                a_ind = torch.randperm(len(valid_count))[:num_comparisons]
                b_ind = torch.randperm(len(valid_count))[:num_comparisons]

                # create the positive trajectory:
                # 1. the trajectories that are less intervened
                # 2. the trajectories that use human actions
                # To do so, a quick workaround is for each episode select the less intervened chunks and form a pos pool.
                # print(1111)

                a_count = valid_count[a_ind]
                a_obs = obs[a_ind]
                a_actions_behavior = actions_behavior[a_ind]
                a_actions_novice = actions_novice[a_ind]

                b_count = valid_count[b_ind]
                b_obs = obs[b_ind]
                b_actions_behavior = actions_behavior[b_ind]
                b_actions_novice = actions_novice[b_ind]

                b_pref_a_label = b_count > a_count

                # If traj b > traj a, traj b is positive and actions should be behavior. Otherwise should use novice.
                b_actions = torch.where(b_pref_a_label[:, None, None], b_actions_behavior, b_actions_novice)
                a_actions = torch.where(b_pref_a_label[:, None, None], a_actions_novice, a_actions_behavior)

                flatten_obs = torch.cat([a_obs.flatten(0, 1), b_obs.flatten(0, 1)], dim=0)
                flatten_actions = torch.cat([a_actions.flatten(0, 1), b_actions.flatten(0, 1)], dim=0)


                _, log_probs, _ = self.policy.evaluate_actions(flatten_obs, flatten_actions)
                a_log_probs, b_log_probs = torch.chunk(log_probs, 2)
                a_log_probs = a_log_probs.reshape(num_comparisons, num_comparisons)
                b_log_probs = b_log_probs.reshape(num_comparisons, num_comparisons)


                # For debug:
                # gt_a = torch.stack(
                #     [self.policy.evaluate_actions(a_obs[i], a_actions[i])[1] for i in range(num_comparisons)]
                # )
                # gt_b = torch.stack(
                #     [self.policy.evaluate_actions(b_obs[i], b_actions[i])[1] for i in range(num_comparisons)]
                # )
                # va = a_log_probs - gt_a
                # vb = b_log_probs - gt_b
                # print(222)

                adv_a = alpha * a_log_probs
                adv_b = alpha * b_log_probs
                adv_a = adv_a.sum(dim=-1)
                adv_b = adv_b.sum(dim=-1)


                # TODO: Grid search the bias factor.
                b_pref_a_label_soft = b_pref_a_label.float()
                b_pref_a_label_soft[b_count == a_count] = 0.5
                cpl_loss, accuracy = biased_bce_with_logits(adv_a, adv_b, b_pref_a_label_soft, bias=0.5)

                stat_recorder["adv_pos"].append(torch.where(b_pref_a_label, adv_b, adv_a).mean().item())
                stat_recorder["adv_neg"].append(torch.where(~b_pref_a_label, adv_b, adv_a).mean().item())
                stat_recorder["int_count_pos"].append(torch.where(b_pref_a_label, b_count, a_count).float().mean().item())
                stat_recorder["int_count_neg"].append(torch.where(~b_pref_a_label, b_count, a_count).float().mean().item())

            else:
                _, log_prob_human_tmp, _ = self.policy.evaluate_actions(
                    obs[interventions], actions_behavior[interventions]
                )
                log_prob_human = log_prob_human_tmp.new_zeros([bs, max_s])
                log_prob_human[interventions] = log_prob_human_tmp

                _, log_prob_agent_tmp, _ = self.policy.evaluate_actions(
                    obs[interventions], actions_novice[interventions]
                )
                log_prob_agent = log_prob_agent_tmp.new_zeros([bs, max_s])
                log_prob_agent[interventions] = log_prob_agent_tmp

                adv_human = alpha * log_prob_human
                adv_agent = alpha * log_prob_agent
                adv_human = adv_human.sum(dim=-1)
                adv_agent = adv_agent.sum(dim=-1)

                # If label = 1, then adv_human > adv_agent
                label = torch.ones_like(adv_human)
                cpl_loss, accuracy = biased_bce_with_logits(adv_agent, adv_human, label.float(), bias=0.5)

                stat_recorder["adv_pos"].append(adv_human.mean().item())
                stat_recorder["adv_neg"].append(adv_agent.mean().item())

            # TODO: Compared to PVP, we remove TD loss here.
            # Optimize the critics
            # critic_loss = cpl_loss
            # if critic_loss is not None:
            #     self.critic.optimizer.zero_grad()
            #     critic_loss.backward()
            #     self.critic.optimizer.step()

            # Optimization step
            self.policy.optimizer.zero_grad()
            cpl_loss.backward()
            # Clip grad norm
            # th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            # Stats
            stat_recorder["cpl_loss"].append(cpl_loss.item() if cpl_loss is not None else float('nan'))
            stat_recorder["cpl_accuracy"].append(accuracy.item() if accuracy is not None else float('nan'))

            # Delayed policy updates
            # if self._n_updates % self.policy_delay == 0:
            #     # Compute actor loss
            #     obs = torch.concatenate([ep.observations for ep in replay_data_agent], dim=0)
            #     action = torch.concatenate([ep.actions_behavior for ep in replay_data_agent], dim=0)
            #
            #
            #     # TODO: As the value is interpreted as advantage, maybe we should use policy gradient here?
            #     # actor_loss = -self.critic.q1_forward(obs, self.actor(obs)).mean()
            #
            #     # Policy gradient:
            #     # action = self.policy.actor(obs)
            #     adv = self.critic.q1_forward(obs, action)
            #     mean, log_std, _ = self.policy.actor.get_action_dist_params(obs)
            #     dist = self.policy.actor.action_dist.proba_distribution(mean, log_std)
            #     log_prob_human = dist.log_prob(action)  #.sum(dim=-1)  # Don't do the sum.
            #     actor_loss = - (adv * log_prob_human).mean()
            #
            #
            #     # Optimize the actor
            #     self.actor.optimizer.zero_grad()
            #     actor_loss.backward()
            #     self.actor.optimizer.step()
            #     self.logger.record("train/actor_loss", actor_loss.item())
            #
            #     # polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)  # TODO: not used.
            #     # polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)  # TODO: not used.

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
