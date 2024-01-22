from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from torch.nn import functional as F

from pvp.sb3.common.noise import ActionNoise
from pvp.sb3.common.type_aliases import GymEnv, Schedule
from pvp.sb3.common.utils import polyak_update
from pvp.sb3.haco.haco_buffer import HACOReplayBuffer
from pvp.sb3.haco.policies import HACOPolicy
from pvp.sb3.sac import SAC
from collections import defaultdict


class HACO(SAC):
    def __init__(
        self,
        policy: Union[str, Type[HACOPolicy]],
        env: Union[GymEnv, str],
        learning_rate: dict = dict(actor=0.0, critic=0.0, entropy=0.0),
        buffer_size: int = 100,  # Shrink the size to reduce memory consumption when testing
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[HACOReplayBuffer] = HACOReplayBuffer,  # PZH: !! Use HACO Replay Buffer
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = True,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,

        # PZH: Our new introduce hyper-parameters
        cql_coefficient=1,
        monitor_wrapper=False
    ):

        assert replay_buffer_class == HACOReplayBuffer

        super(HACO, self).__init__(
            policy,
            env,
            HACOPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            monitor_wrapper=monitor_wrapper
        )
        # PZH: Define some new variables
        self.cql_coefficient = cql_coefficient

    def _create_aliases(self) -> None:
        super(HACO, self)._create_aliases()
        self.cost_critic = self.policy.cost_critic
        self.cost_critic_target = self.policy.cost_critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = {
            "actor": self.actor.optimizer,
            "critic": self.critic.optimizer,
            # "cost_critic": self.cost_critic.optimizer
        }
        if self.ent_coef_optimizer is not None:
            optimizers["entropy"] = self.ent_coef_optimizer

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        stat_recorder = defaultdict(list)

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            # ===== Optimizing the entropy coefficient =====
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                # ent_coef_losses.append(ent_coef_loss.item())
                stat_recorder['ent_coef_loss'].append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            stat_recorder["entropy"].append(-log_prob.mean().item())
            # entropys.append(-log_prob.mean().item())
            # ent_coefs.append(ent_coef.item())
            stat_recorder["ent_coef"].append(ent_coef.item())
            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # ===== Optimizing the critic and the cost critic =====
            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)

                # Compute the target Q values
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                # PZH: Compute the target cost Q values
                next_cost_q_values = th.cat(self.cost_critic_target(replay_data.next_observations, next_actions), dim=1)
                next_cost_q_values, _ = th.min(next_cost_q_values, dim=1, keepdim=True)
                # We don't take the entropy into account when computing the next cost Q.
                next_cost_q_values = next_cost_q_values  # - ent_coef * next_log_prob.reshape(-1, 1)
                # PZH: Note that we use
                target_cost_q_values = (
                    replay_data.intervention_costs + (1 - replay_data.dones) * self.gamma * next_cost_q_values
                )

            # === Optimizing the critic ===
            current_q_behavior_values = self.critic(replay_data.observations, replay_data.actions_behavior)
            current_q_novice_values = self.critic(replay_data.observations, replay_data.actions_novice)

            stat_recorder["q_value_behavior"].append(current_q_behavior_values[0].mean().item())
            stat_recorder["q_value_novice"].append(current_q_novice_values[0].mean().item())

            critic_loss = []
            for (current_q_behavior, current_q_novice) in zip(current_q_behavior_values, current_q_novice_values):
                l = 0.5 * F.mse_loss(current_q_behavior, target_q_values)

                # PZH: Here is the CQL loss
                l -= th.mean(replay_data.interventions * self.cql_coefficient * (current_q_behavior - current_q_novice))

                critic_loss.append(l)
            critic_loss = sum(critic_loss)
            # critic_losses.append(critic_loss.item())
            stat_recorder["critic_loss"].append(critic_loss.item())

            # === Optimizing the cost critic ===
            # FIXME(pzh): We use "behavior actions" in old impl. We used "novice actions" at it fails.
            #  Does this point the critical point?????? Should we use "behavior actions"????
            current_cost_q_values = self.cost_critic(replay_data.observations, replay_data.actions_behavior)
            cost_critic_loss = 0.5 * sum(
                [F.mse_loss(current_cost_q, target_cost_q_values) for current_cost_q in current_cost_q_values]
            )
            for i, v in enumerate(current_cost_q_values):
                stat_recorder["cost_q_value_{}".format(i)].append(v.mean().item())
            stat_recorder["cost_critic_loss"].append(cost_critic_loss.item())
            merged_critic_loss = cost_critic_loss + critic_loss

            # ===== Optimizing the actor =====
            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)

            # TODO(pzh): They use q_0's output, not the min
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            stat_recorder["q_value_min"].append(min_qf_pi.mean().item())

            # Compute the cost Q in the actor loss
            cost_q_values_pi = th.cat(self.cost_critic(replay_data.observations, actions_pi), dim=1)

            # TODO(pzh): They use cost_q_0's output, not the min
            min_cost_qf_pi, _ = th.min(cost_q_values_pi, dim=1, keepdim=True)

            stat_recorder["cost_q_value_min"].append(min_cost_qf_pi.mean().item())

            # PZH: Apply the Lagrangian multiplier to the actor loss
            native_actor_loss = ent_coef * log_prob - min_qf_pi
            cost_actor_loss = min_cost_qf_pi
            actor_loss = (native_actor_loss + cost_actor_loss).mean()

            stat_recorder["actor_loss"].append(native_actor_loss.mean().item())
            stat_recorder["cost_actor_loss"].append(cost_actor_loss.mean().item())

            if self.policy_kwargs["share_features_extractor"] == "critic":
                self._optimize_actor(actor_loss=actor_loss)
                self._optimize_critics(merged_critic_loss=merged_critic_loss)
            elif self.policy_kwargs["share_features_extractor"] == "actor":
                raise ValueError()
            else:
                self._optimize_actor(actor_loss=actor_loss)
                self._optimize_critics(merged_critic_loss=merged_critic_loss)

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.cost_critic.parameters(), self.cost_critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates)
        for key, values in stat_recorder.items():
            self.logger.record("train/{}".format(key), np.mean(values))

    def _optimize_actor(self, actor_loss):
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

    def _optimize_critics(self, merged_critic_loss):
        self.critic.optimizer.zero_grad()
        merged_critic_loss.backward()
        self.critic.optimizer.step()

    def _excluded_save_params(self) -> List[str]:
        return super(HACO, self)._excluded_save_params() + ["cost_critic", "cost_critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        # state_dicts = ["policy", "actor.optimizer", "critic.optimizer", "cost_critic.optimizer"]
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables
