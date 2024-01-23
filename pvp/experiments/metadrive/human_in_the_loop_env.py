import copy
from collections import deque

import numpy as np
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.engine.core.onscreen_message import ScreenMessage
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.policy.manual_control_policy import TakeoverPolicy
from metadrive.utils.math import safe_clip

ScreenMessage.SCALE = 0.1


class HumanInTheLoopEnv(SafeMetaDriveEnv):
    """
    This Env depends on the new version of MetaDrive
    """

    _takeover_recorder = deque(maxlen=2000)

    total_steps = 0

    def default_config(self):
        config = super(HumanInTheLoopEnv, self).default_config()
        config.update(
            {
                # TODO: add more docs for the config here.
                "num_scenarios": 50,
                "start_seed": 100,

                # Environment setting:

                # Controller:

                # Reward and cost setting:
                "cost_to_reward": True,
                "traffic_density": 0.06,
                "manual_control": False,
                "out_of_route_done": True,
                "controller": "keyboard",  # Selected from [keyboard, xbox, steering_wheel].
                "agent_policy": TakeoverPolicy,
                "only_takeover_start_cost": True,
                "main_exp": True,
                "random_spawn": True,
                "cos_similarity": True,
                "in_replay": False,

                # Visualization
                "vehicle_config": {
                    "show_dest_mark": True,
                    "show_line_to_dest": True,
                    "show_line_to_navi_mark": True,
                }
            },
            allow_add_new_key=True
        )
        return config

    def reset(self, *args, **kwargs):
        self.in_stop = False
        self.t_o = False
        self.total_takeover_cost = 0
        self.input_action = None
        obs, info = super(HumanInTheLoopEnv, self).reset(*args, **kwargs)
        if self.config["random_spawn"]:
            self.config["vehicle_config"]["spawn_lane_index"] = (
                FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, self.engine.np_random.randint(3)
            )

        # The training code is for older version of gym, so we discard the additional info from the reset.
        return obs

    def _get_step_return(self, actions, engine_info):
        o, r, tm, tc, engine_info = super(HumanInTheLoopEnv, self)._get_step_return(actions, engine_info)

        # TODO(pzh): Double check what is this
        # if self.config["in_replay"]:
        #     return o, r, d, engine_info

        d = tm or tc

        controller = self.engine.get_policy(self.vehicle.id)
        last_t = self.t_o
        self.t_o = controller.takeover if hasattr(controller, "takeover") else False
        engine_info["takeover_start"] = True if not last_t and self.t_o else False
        engine_info["takeover"] = self.t_o and not engine_info["takeover_start"]
        condition = engine_info["takeover_start"] if self.config["only_takeover_start_cost"] else self.t_o

        if not condition:
            self.total_takeover_cost += 0
            engine_info["takeover_cost"] = 0
        else:
            cost = self.get_takeover_cost(engine_info)
            self.total_takeover_cost += cost
            engine_info["takeover_cost"] = cost

        engine_info["total_takeover_cost"] = self.total_takeover_cost
        engine_info["native_cost"] = engine_info["cost"]
        engine_info["total_native_cost"] = self.episode_cost
        return o, r, d, engine_info

    def _is_out_of_road(self, vehicle):
        ret = (not vehicle.on_lane) or vehicle.crash_sidewalk
        if self.config["out_of_route_done"]:
            ret = ret or vehicle.out_of_route
        return ret

    def step(self, actions):
        self.input_action = copy.copy(actions)
        ret = super(HumanInTheLoopEnv, self).step(actions)
        while self.in_stop:
            self.engine.taskMgr.step()

        self._takeover_recorder.append(self.t_o)
        if self.config["use_render"] and self.config["main_exp"] and not self.config["in_replay"]:
            super(HumanInTheLoopEnv, self).render(
                text={
                    "Total Cost": round(self.episode_cost, 3),
                    "Takeover Cost": round(self.total_takeover_cost, 3),
                    "Takeover": self.t_o,
                    "COST": ret[-1]["takeover_cost"],
                    "Total Step": self.total_steps,
                    "Takeover Rate": "{:.3f}%".format(np.mean(np.array(self._takeover_recorder) * 100)),
                    "Pause (Press E)": "",
                }
            )

        self.total_steps += 1

        return ret

    def stop(self):
        self.in_stop = not self.in_stop

    def setup_engine(self):
        super(HumanInTheLoopEnv, self).setup_engine()
        self.engine.accept("e", self.stop)

    def get_takeover_cost(self, info):
        if not self.config["cos_similarity"]:
            return 1

        takeover_action = safe_clip(np.array(info["raw_action"]), -1, 1)
        agent_action = safe_clip(np.array(self.input_action), -1, 1)

        multiplier = (agent_action[0] * takeover_action[0] + agent_action[1] * takeover_action[1])
        divident = np.linalg.norm(takeover_action) * np.linalg.norm(agent_action)
        if divident < 1e-6:
            cos_dist = 1.0
        else:
            cos_dist = multiplier / divident

        return 1 - cos_dist


if __name__ == "__main__":
    env = HumanInTheLoopEnv(
        {
            "manual_control": True,
            "disable_model_compression": True,
            "use_render": True,
            "main_exp": True
        }
    )
    env.reset()
    while True:
        _, _, done, _ = env.step([0, 0])
        if done:
            env.reset()
