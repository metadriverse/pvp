import copy
import math
import torch
from metadrive.engine.engine_utils import get_global_config
from metadrive.engine.logger import get_logger
from metadrive.examples.ppo_expert.numpy_expert import ckpt_path
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.policy.env_input_policy import EnvInputPolicy
import numpy as np
from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
import pathlib

FOLDER_PATH = pathlib.Path(__file__).parent

logger = get_logger()


def get_expert():

    from pvp.sb3.common.save_util import load_from_zip_file
    from pvp.sb3.ppo import PPO
    from pvp.sb3.ppo.policies import ActorCriticPolicy

    train_env = HumanInTheLoopEnv(config={'manual_control': False, "use_render": False})

    # Initialize agent
    algo_config = dict(
        policy=ActorCriticPolicy,
        n_steps=1024,  # n_steps * n_envs = total_batch_size
        n_epochs=20,
        learning_rate=5e-5,
        batch_size=256,
        clip_range=0.1,
        vf_coef=0.5,
        ent_coef=0.0,
        max_grad_norm=10.0,
        # tensorboard_log=trial_dir,
        create_eval_env=False,
        verbose=2,
        # seed=seed,
        device="auto",
        env=train_env
    )
    model = PPO(**algo_config)

    ckpt = FOLDER_PATH / "metadrive_pvp_20m_steps"

    print(f"Loading checkpoint from {ckpt}!")
    data, params, pytorch_variables = load_from_zip_file(ckpt, device=model.device, print_system_info=False)
    model.set_parameters(params, exact_match=True, device=model.device)
    print(f"Model is loaded from {ckpt}!")

    train_env.close()

    return model.policy


def obs_correction(obs):
    # due to coordinate correction, this observation should be reversed
    obs[15] = 1 - obs[15]
    obs[10] = 1 - obs[10]
    return obs


def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2 * math.pi * var)**.5
    num = math.exp(-(float(x) - float(mean))**2 / (2 * var))
    return num / denom


def load():
    global _expert_weights
    if _expert_weights is None:
        _expert_weights = np.load(ckpt_path)
    return _expert_weights


_expert = get_expert()


class FakeHumanEnv(HumanInTheLoopEnv):
    last_takeover = None
    last_obs = None
    expert = None

    def default_config(self):
        """Revert to use the RL policy (so no takeover signal will be issued from the human)"""
        config = super(FakeHumanEnv, self).default_config()
        config.update(
            {
                "agent_policy": EnvInputPolicy,
                "free_level": 0.95,
                "manual_control": False,
                "use_render": False,

                "expert_deterministic": False,
            }
        )
        return config

    def step(self, actions):
        """Compared to the original one, we call expert_action_prob here and implement a takeover function."""
        actions = np.asarray(actions).astype(np.float32)
        self.agent_action = copy.copy(actions)
        self.last_takeover = self.takeover

        # ===== Get expert action and determine whether to take over! =====
        if self.expert is None:
            global _expert
            self.expert = _expert
        last_obs, _ = self.expert.obs_to_tensor(self.last_obs)
        distribution = self.expert.get_distribution(last_obs)
        log_prob = distribution.log_prob(torch.from_numpy(actions).to(last_obs.device))
        action_prob = log_prob.exp().detach().cpu().numpy()

        if self.config["expert_deterministic"]:
            expert_action = distribution.mode().detach().cpu().numpy()
        else:
            expert_action = distribution.sample().detach().cpu().numpy()

        assert expert_action.shape[0] == action_prob.shape[0] == 1
        action_prob = action_prob[0]
        expert_action = expert_action[0]
        if action_prob < 1 - self.config['free_level']:

            # print(f"Action probability: {action_prob}, agent action: {actions}, expert action: {expert_action},")

            actions = expert_action
            self.takeover = True
        else:
            self.takeover = False
        # print(f"Action probability: {action_prob:.3f}, agent action: {actions}, expert action: {expert_action}, takeover: {self.takeover}")

        o, r, d, i = super(HumanInTheLoopEnv, self).step(actions)
        self.takeover_recorder.append(self.takeover)
        self.total_steps += 1

        i["takeover_log_prob"] = log_prob.item()

        if self.config["use_render"]:  # and self.config["main_exp"]: #and not self.config["in_replay"]:
            super(HumanInTheLoopEnv, self).render(
                text={
                    "Total Cost": round(self.total_cost, 2),
                    "Takeover Cost": round(self.total_takeover_cost, 2),
                    "Takeover": "TAKEOVER" if self.takeover else "NO",
                    "Total Step": self.total_steps,
                    # "Total Time": time.strftime("%M:%S", time.gmtime(time.time() - self.start_time)),
                    "Takeover Rate": "{:.2f}%".format(np.mean(np.array(self.takeover_recorder) * 100)),
                    "Pause": "Press E",
                }
            )

        assert i["takeover"] == self.takeover
        return o, r, d, i

    def _get_step_return(self, actions, engine_info):
        """Compared to original one, here we don't call expert_policy, but directly get self.last_takeover."""
        o, r, tm, tc, engine_info = super(HumanInTheLoopEnv, self)._get_step_return(actions, engine_info)
        self.last_obs = o
        d = tm or tc
        last_t = self.last_takeover
        engine_info["takeover_start"] = True if not last_t and self.takeover else False
        engine_info["takeover"] = self.takeover
        condition = engine_info["takeover_start"] if self.config["only_takeover_start_cost"] else self.takeover
        if not condition:
            engine_info["takeover_cost"] = 0
        else:
            cost = self.get_takeover_cost(engine_info)
            self.total_takeover_cost += cost
            engine_info["takeover_cost"] = cost
        engine_info["total_takeover_cost"] = self.total_takeover_cost
        engine_info["native_cost"] = engine_info["cost"]
        engine_info["episode_native_cost"] = self.episode_cost
        self.total_cost += engine_info["cost"]
        self.total_takeover_count += 1 if self.takeover else 0
        engine_info["total_takeover_count"] = self.total_takeover_count
        engine_info["total_cost"] = self.total_cost
        # engine_info["total_cost_so_far"] = self.total_cost
        return o, r, d, engine_info

    def _get_reset_return(self, reset_info):
        o, info = super(HumanInTheLoopEnv, self)._get_reset_return(reset_info)
        self.last_obs = o
        self.last_takeover = False
        return o, info


if __name__ == "__main__":
    env = FakeHumanEnv(dict(free_level=0.95, use_render=False))
    env.reset()
    while True:
        _, _, done, info = env.step([0, 1])
        # done = tm or tc
        # env.render(mode="topdown")
        if done:
            print(info)
            env.reset()
