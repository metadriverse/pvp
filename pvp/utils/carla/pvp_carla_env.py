import copy
import platform
import time
from collections import defaultdict
from collections import deque

import gym
import numpy as np
from easydict import EasyDict

from pvp_iclr_release.utils.print_dict_utils import merge_dicts
from pvp_iclr_release.utils.carla.core.envs.simple_carla_env import SimpleCarlaEnv
from pvp_iclr_release.utils.carla.demo.simple_rl.env_wrapper import ContinuousBenchmarkEnvWrapper
from pvp_iclr_release.utils.carla.demo.simple_rl.sac_train import compile_config
from pvp_iclr_release.utils.human_interface import SteeringWheelController
from pvp_iclr_release.utils.older_utils import ForceFPS, merge_dicts

is_windows = "Win" in platform.system()


def safe_clip(array, min_val, max_val):
    array = np.nan_to_num(array.astype(np.float64), copy=False, nan=0.0, posinf=max_val, neginf=min_val)
    return np.clip(array, min_val, max_val).astype(np.float64)


train_config = dict(

    obs_mode="birdview",
    force_fps=0,
    disable_vis=True,
    debug_vis=False,
    port=9000,
    disable_takeover=False,
    show_text=True,
    normalize_obs=False,

    env=dict(
        collector_env_num=1,
        evaluator_env_num=0,
        simulator=dict(
            town='Town01',
            disable_two_wheels=True,
            verbose=False,
            waypoint_num=32,
            planner=dict(
                type='behavior',
                resolution=1,
            ),
            # obs = ..., We autofill this in PVPEnv
        ),
        col_is_failure=True,
        stuck_is_failure=False,
        wrong_direction_is_failure=False,
        off_route_is_failure=True,
        off_road_is_failure=True,
        ignore_light=True,

        off_route_distance=10.0,

        visualize=dict(
            type="vis",
            outputs=['show'],
            location="center"
        ),
        # visualize=None,
        manager=dict(
            collect=dict(
                auto_reset=True,
                shared_memory=False,
                context='spawn',
                max_retry=1,
            ),
            eval=dict()
        ),
        wrapper=dict(
            # Collect and eval suites for training
            collect=dict(suite='NoCrashTown01-v1'),

        ),
    ),
)


class PVPEnv(ContinuousBenchmarkEnvWrapper):
    def __init__(self, config=None, ):
        main_config = copy.deepcopy(train_config)
        disable_vis = config.get("disable_vis", None)
        if disable_vis is None:
            disable_vis = main_config["disable_vis"]
        debug_vis = config.get("debug_vis", None)
        if debug_vis is None:
            debug_vis = main_config["debug_vis"]
        if disable_vis or debug_vis:
            sensors = []
        else:
            sensors = [dict(
                name='vis',
                type='rgb',
                size=[2100, 1400],
                position=[-5.5, 0, 2.8],
                rotation=[-15, 0, 0],
            )]

        if config["obs_mode"] == "birdview":
            sensors.append(
                dict(
                    name="birdview",
                    type='bev',
                    size=[84, 84],
                    pixels_per_meter=2,
                    pixels_ahead_vehicle=16,
                )
            )
            if debug_vis:
                main_config["env"]["visualize"]["type"] = "birdview"
        elif config["obs_mode"] in ["first", "firststack"]:
            sensors.append(
                dict(
                    name='rgb',
                    type='rgb',
                    size=[320, 180],

                    # xxx note: Default is:
                    # 'position':[2.0, 0.0, 1.4],
                    # 'rotation':[0, 0, 0],
                )
            )
            if config["obs_mode"] == "firststack":
                self._frame_stack = deque(maxlen=5)
            if debug_vis:
                main_config["env"]["visualize"]["type"] = "rgb"
        elif config["obs_mode"] == "birdview42":
            sensors.append(
                dict(
                    name="birdview",
                    type='bev',
                    size=[42, 42],
                    pixels_per_meter=6
                )
            )
            if debug_vis:
                main_config["env"]["visualize"]["type"] = "birdview"
        else:
            raise ValueError("Unknown obs mode: {}".format(config["obs_mode"]))
        main_config["env"]["simulator"]["obs"] = tuple(sensors)
        main_config["env"]["obs_mode"] = config["obs_mode"]
        main_config = EasyDict(main_config)
        if config is not None and config:
            main_config = EasyDict(merge_dicts(main_config, config))
        if main_config["show_text"] is False:
            main_config["env"]["visualize"]["show_text"] = False
        if main_config["disable_vis"]:
            main_config["env"]["visualize"] = None

        # self.eval = eval
        # if eval:
        #     train_config["env"]["wrapper"]["collect"]["suite"] = 'FullTown02-v1'
        #     raise ValueError("Not sure what this is doing.")
        cfg = compile_config(main_config)

        port = cfg["port"]
        disable_takeover = cfg["disable_takeover"]

        # TODO(xxx): This is really stupid. The config system need more consideration!
        self.xxx_cfg = cfg
        super(PVPEnv, self).__init__(SimpleCarlaEnv(cfg.env, "localhost", port, None), cfg.env.wrapper.collect)

        # xxx: We should not escape this error.
        # try:
        if disable_takeover:
            self.controller = None
        else:
            self.controller = SteeringWheelController(disable_takeover)
        # except:
        #     self.controller = None

        self.last_takeover = False

        self.episode_recorder = defaultdict(list)

        # TODO(xxx): I hard coded here!
        force_fps = self.xxx_cfg["force_fps"]
        self.force_fps = ForceFPS(fps=force_fps, start=(force_fps > 0))
        self.normalize_obs = self.xxx_cfg["normalize_obs"]

    def step(self, action):
        if self.controller is not None:
            human_action = self.controller.process_input(self.env._simulator_databuffer['state']['speed_kmh'])
            takeover = self.controller.left_shift_paddle or self.controller.right_shift_paddle
        else:
            human_action = [0, 0]
            takeover = False

        # print("CURRENT TAKEOVER: ", takeover)

        o, r, d, info = super(PVPEnv, self).step(human_action if takeover else action)

        self.episode_recorder["reward"].append(r)

        # if not self.last_takeover and takeover:
        #     cost = self.get_takeover_cost(human_action, action)
        #     self.episode_recorder["cost"].append(cost)
        #     info["takeover_cost"] = cost
        # else:
        #     self.episode_recorder["cost"].append(0)
        #     info["takeover_cost"] = 0
        #
        # info["takeover"] = takeover
        # info["takeover_start"] = takeover

        info["takeover_start"] = True if not self.last_takeover and takeover else False
        info["takeover"] = takeover and not info["takeover_start"]
        condition = info["takeover_start"]

        if not condition:
            # self.total_takeover_cost += 0
            self.episode_recorder["cost"].append(0)
            info["takeover_cost"] = 0
        else:
            cost = self.get_takeover_cost(human_action, action)
            self.episode_recorder["cost"].append(cost)
            # self.total_takeover_cost += cost
            info["takeover_cost"] = cost

        # info["total_takeover_cost"] = self.total_takeover_cost

        info["total_cost"] = info["total_takeover_cost"] = sum(self.episode_recorder["cost"])
        info["raw_action"] = list(action) if not takeover else list(human_action)
        self.last_takeover = takeover

        info["velocity"] = self.env._simulator_databuffer['state']['speed']
        self.episode_recorder["velocity"].append(info["velocity"])

        info["steering"] = float(info["raw_action"][0])
        info["acceleration"] = float(info["raw_action"][1])
        info["step_reward"] = float(r)
        info["cost"] = float(self.native_cost(info))
        info["native_cost"] = float(info["cost"])
        info["out_of_road"] = info["off_road"]
        info["crash"] = info["collided"]
        info["arrive_dest"] = info["success"]
        info["episode_length"] = info["tick"]
        info["episode_reward"] = float(sum(self.episode_recorder["reward"]))
        info["episode_average_velocity"] = (
                sum(self.episode_recorder["velocity"]) / len(self.episode_recorder["velocity"])
        )
        # if not self.eval:

        if not self.xxx_cfg["disable_vis"]:
            self.render()

        self.force_fps.sleep_if_needed()
        # print({k: (v.shape, v.dtype) for k, v in o.items()})

        return o, float(r[0]), d, info

        # return self.observation_space.sample(), 0.0, False, {}

    def native_cost(self, info):
        if info["off_route"] or info["off_road"] or info["collided"] or info["wrong_direction"]:
            return 1
        else:
            return 0

    def get_takeover_cost(self, human_action, agent_action):
        takeover_action = safe_clip(np.array(human_action), -1, 1)
        agent_action = safe_clip(np.array(agent_action), -1, 1)
        # cos_dist = (agent_action[0] * takeover_action[0] + agent_action[1] * takeover_action[1]) / 1e-6 +(
        #         np.linalg.norm(takeover_action) * np.linalg.norm(agent_action))

        multiplier = (agent_action[0] * takeover_action[0] + agent_action[1] * takeover_action[1])
        divident = np.linalg.norm(takeover_action) * np.linalg.norm(agent_action)
        if divident < 1e-6:
            cos_dist = 1.0
        else:
            cos_dist = multiplier / divident
        return 1 - cos_dist

    def reset(self, *args, **kwargs):
        self.last_takeover = False
        self.episode_recorder.clear()

        if self.controller:
            self.controller.reset()
        obs = super(PVPEnv, self).reset()

        self.force_fps.clear()

        return obs

    def postprocess_obs(self, raw_obs, in_reset=False):
        speed = float(raw_obs["speed_kmh"])  # in km/h
        normalized_speed = min(speed, self.env._max_speed_kmh) / self.env._max_speed_kmh
        if self.xxx_cfg["obs_mode"] in ["birdview", "birdview42"]:
            obs = raw_obs["birdview"][..., [0, 1, 5, 6, 8]]
            if self.normalize_obs:
                obs = obs.astype(np.float32) / 255
            obs_out = {
                "image": obs,
                'speed': np.array([normalized_speed]),
            }
        elif self.xxx_cfg["obs_mode"] == "first":
            obs = raw_obs["rgb"]
            if self.normalize_obs:
                obs = obs.astype(np.float32) / 255
            obs_out = {
                'image': obs,
                'speed': np.array([normalized_speed]),
            }
        elif self.xxx_cfg["obs_mode"] == "firststack":
            obs = raw_obs['rgb']
            obs = np.mean(obs, axis=-1)
            if in_reset:
                for _ in range(self._frame_stack.maxlen):
                    self._frame_stack.append(obs)
            else:
                self._frame_stack.append(obs)
            obs_out = np.stack(list(self._frame_stack), axis=-1).astype(np.uint8)
            if self.normalize_obs:
                obs_out = obs_out.astype(np.float32) / 255
        else:
            raise ValueError()
        return obs_out

    @property
    def action_space(self):
        return gym.spaces.Box(-1.0, 1.0, shape=(2,))

    @property
    def observation_space(self):
        if self.xxx_cfg["obs_mode"] == "birdview":
            return gym.spaces.Dict({
                "image": gym.spaces.Box(low=0, high=255, shape=(84, 84, 5), dtype=np.uint8),
                "speed": gym.spaces.Box(0., 1.0, shape=(1,))
            })
        elif self.xxx_cfg["obs_mode"] == "first":
            return gym.spaces.Dict({
                "image": gym.spaces.Box(low=0, high=255, shape=(180, 320, 3), dtype=np.uint8),
                "speed": gym.spaces.Box(0., 1.0, shape=(1,))
            })
        elif self.xxx_cfg["obs_mode"] == "firststack":
            return gym.spaces.Box(low=0, high=255, shape=(180, 320, 5), dtype=np.uint8)

        elif self.xxx_cfg["obs_mode"] == "birdview42":
            return gym.spaces.Dict({
                "image": gym.spaces.Box(low=0, high=255, shape=(42, 42, 5), dtype=np.uint8),
                "speed": gym.spaces.Box(0., 1.0, shape=(1,))
            })
        else:
            raise ValueError("Wrong obs_mode")

    def render(self):
        return super(PVPEnv, self).render(takeover=self.last_takeover)


if __name__ == "__main__":
    # from pvp_iclr_release.stable_baseline3.common.env_checker import check_env
    # env = PVPEnv({"obs_mode": "birdview", "disable_vis": False})
    # check_env(env)
    # env.close()

    env = PVPEnv({
        "obs_mode": "firststack",
        "force_fps": 0,
        "disable_vis": False,
        "debug_vis": True,
        "disable_takeover": False,

        # "env": {"visualize": {"location": "upper left"}}
    })
    env.seed(0)
    o = env.reset()

    st = time.time()
    count = 0
    while True:
        # if not env.observation_space.contains(o):
        #     print(
        #         "Wrong observation and space: ",
        #         {k: (v.shape, v.dtype) for k, v in env.observation_space.spaces.items()},
        #         {k: (v.shape, v.dtype) for k, v in o.items()}
        #     )
        o, r, d, i = env.step([0., 1.0])
        count += 1

        # if count == 100:
        #     import pickle
        #
        #     with open("first_obs_sample.pkl", "wb") as f:
        #         pickle.dump(o, f)
        #     break

        ct = time.time()
        fps = 1 / (ct - st)
        st = ct
        # print("RL steps per second: ", fps)

        if d:
            env.reset()
