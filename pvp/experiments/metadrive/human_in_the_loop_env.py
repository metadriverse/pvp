import copy
import time
from collections import deque

import numpy as np
from metadrive.engine.core.onscreen_message import ScreenMessage
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.policy.manual_control_policy import TakeoverPolicyWithoutBrake
from metadrive.utils.math import safe_clip

ScreenMessage.SCALE = 0.1

HUMAN_IN_THE_LOOP_ENV_CONFIG = {
    # Environment setting:
    "out_of_route_done": True,  # Raise done if out of route.
    "num_scenarios": 50,  # There are totally 50 possible maps.
    "start_seed": 100,  # We will use the map 100~150 as the default training environment.
    "traffic_density": 0.06,

    # Reward and cost setting:
    "cost_to_reward": True,  # Cost will be negated and added to the reward. Useless in PVP.
    "cos_similarity": False,  # If True, the takeover cost will be the cos sim between a_h and a_n. Useless in PVP.

    # Set up the control device. Default to use keyboard with the pop-up interface.
    "manual_control": True,
    "agent_policy": TakeoverPolicyWithoutBrake,
    "controller": "keyboard",  # Selected from [keyboard, xbox, steering_wheel].
    "only_takeover_start_cost": False,  # If True, only return a cost when takeover starts. Useless in PVP.

    # Visualization
    "vehicle_config": {
        "show_dest_mark": True,  # Show the destination in a cube.
        "show_line_to_dest": True,  # Show the line to the destination.
        "show_line_to_navi_mark": True,  # Show the line to next navigation checkpoint.
    }
}


class HumanInTheLoopEnv(SafeMetaDriveEnv):
    """
    Human-in-the-loop Env Wrapper for the Safety Env in MetaDrive.
    Add code for computing takeover cost and add information to the interface.
    """
    total_steps = 0
    total_takeover_cost = 0
    total_cost = 0
    takeover = False
    takeover_recorder = deque(maxlen=2000)
    agent_action = None
    in_pause = False
    start_time = time.time()

    def default_config(self):
        config = super(HumanInTheLoopEnv, self).default_config()
        config.update(HUMAN_IN_THE_LOOP_ENV_CONFIG, allow_add_new_key=True)
        return config

    def reset(self, *args, **kwargs):
        self.takeover = False
        self.agent_action = None
        obs, info = super(HumanInTheLoopEnv, self).reset(*args, **kwargs)
        # The training code is for older version of gym, so we discard the additional info from the reset.
        return obs

    def _get_step_return(self, actions, engine_info):
        """Compute takeover cost here."""
        o, r, tm, tc, engine_info = super(HumanInTheLoopEnv, self)._get_step_return(actions, engine_info)
        d = tm or tc

        shared_control_policy = self.engine.get_policy(self.agent.id)
        last_t = self.takeover
        self.takeover = shared_control_policy.takeover if hasattr(shared_control_policy, "takeover") else False
        engine_info["takeover_start"] = True if not last_t and self.takeover else False
        engine_info["takeover"] = self.takeover and not engine_info["takeover_start"]
        condition = engine_info["takeover_start"] if self.config["only_takeover_start_cost"] else self.takeover
        if not condition:
            engine_info["takeover_cost"] = 0
        else:
            cost = self.get_takeover_cost(engine_info)
            self.total_takeover_cost += cost
            engine_info["takeover_cost"] = cost
        engine_info["total_takeover_cost"] = self.total_takeover_cost
        engine_info["native_cost"] = engine_info["cost"]
        engine_info["total_native_cost"] = self.episode_cost
        self.total_cost += engine_info["cost"]

        return o, r, d, engine_info

    def _is_out_of_road(self, vehicle):
        """Out of road condition"""
        ret = (not vehicle.on_lane) or vehicle.crash_sidewalk
        if self.config["out_of_route_done"]:
            ret = ret or vehicle.out_of_route
        return ret

    def step(self, actions):
        """Add additional information to the interface."""
        self.agent_action = copy.copy(actions)
        ret = super(HumanInTheLoopEnv, self).step(actions)
        while self.in_pause:
            self.engine.taskMgr.step()

        self.takeover_recorder.append(self.takeover)
        if self.config["use_render"]:  # and self.config["main_exp"]: #and not self.config["in_replay"]:
            super(HumanInTheLoopEnv, self).render(
                text={
                    "Total Cost": round(self.total_cost, 2),
                    "Takeover Cost": round(self.total_takeover_cost, 2),
                    "Takeover": "TAKEOVER" if self.takeover else "NO",
                    "Total Step": self.total_steps,
                    "Total Time": time.strftime("%M:%S", time.gmtime(time.time() - self.start_time)),
                    "Takeover Rate": "{:.2f}%".format(np.mean(np.array(self.takeover_recorder) * 100)),
                    "Pause": "Press E",
                }
            )

        self.total_steps += 1

        return ret

    def stop(self):
        """Toggle pause."""
        self.in_pause = not self.in_pause

    def setup_engine(self):
        """Introduce additional key 'e' to the interface."""
        super(HumanInTheLoopEnv, self).setup_engine()
        self.engine.accept("e", self.stop)

    def get_takeover_cost(self, info):
        """Return the takeover cost when intervened."""
        if not self.config["cos_similarity"]:
            return 1
        takeover_action = safe_clip(np.array(info["raw_action"]), -1, 1)
        agent_action = safe_clip(np.array(self.agent_action), -1, 1)
        multiplier = (agent_action[0] * takeover_action[0] + agent_action[1] * takeover_action[1])
        divident = np.linalg.norm(takeover_action) * np.linalg.norm(agent_action)
        if divident < 1e-6:
            cos_dist = 1.0
        else:
            cos_dist = multiplier / divident
        return 1 - cos_dist


if __name__ == "__main__":
    env = HumanInTheLoopEnv({
        "manual_control": True,
        "use_render": True,
    })
    env.reset()
    while True:
        _, _, done, _ = env.step([0, 0])
        if done:
            env.reset()
