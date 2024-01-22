from typing import Optional

from pvp.sb3.common.policies import ContinuousCritic
from pvp.sb3.common.torch_layers import BaseFeaturesExtractor
from pvp.sb3.common.type_aliases import Schedule
from pvp.sb3.sac.policies import SACPolicy

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class HACOPolicy(SACPolicy):
    def _build_share_feature_trained_by_actor(self):
        self.actor = self.make_actor()
        actor_parameters = self.actor.parameters()
        self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
        critic_parameters = [
            param for name, param in self.critic.named_parameters() if "features_extractor" not in name
        ]
        return actor_parameters, critic_parameters

    def _build_share_feature_trained_by_critic(self):
        self.critic = self.make_critic()
        critic_parameters = self.critic.parameters()
        self.actor = self.make_actor(features_extractor=self.critic.features_extractor)
        actor_parameters = [param for name, param in self.actor.named_parameters() if "features_extractor" not in name]
        return actor_parameters, critic_parameters

    def _build_independent_feature(self):
        self.actor = self.make_actor()
        self.critic = self.make_critic()
        return self.actor.parameters(), self.critic.parameters()

    def _build(self, lr_schedule: Schedule) -> None:
        assert self.share_features_extractor in ["critic", "actor", False, None]

        if self.share_features_extractor == "actor":
            actor_parameters, critic_parameters = self._build_share_feature_trained_by_actor()
        elif self.share_features_extractor == "critic":
            actor_parameters, critic_parameters = self._build_share_feature_trained_by_critic()
        else:
            actor_parameters, critic_parameters = self._build_independent_feature()

        self.actor.optimizer = self.optimizer_class(
            actor_parameters, lr=lr_schedule["actor"](1), **self.optimizer_kwargs
        )

        # Build cost critic and target cost critic
        self.cost_critic = self.make_cost_critic(features_extractor=self.critic.features_extractor)
        cost_critic_parameters = [
            param for name, param in self.cost_critic.named_parameters() if "features_extractor" not in name
        ]
        critic_parameters = list(critic_parameters) + cost_critic_parameters

        # Critic target should not share the features extractor with critic
        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = self.optimizer_class(
            critic_parameters, lr=lr_schedule["critic"](1), **self.optimizer_kwargs
        )
        self.cost_critic_target = self.make_cost_critic(features_extractor=None)
        self.cost_critic_target.load_state_dict(self.cost_critic.state_dict())

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)
        self.cost_critic_target.set_training_mode(False)

    # PZH: Add a critic to evaluate cost
    def make_cost_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def set_training_mode(self, mode: bool) -> None:
        self.cost_critic.set_training_mode(mode)
        super(HACOPolicy, self).set_training_mode(mode)
