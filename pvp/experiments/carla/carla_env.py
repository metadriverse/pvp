import copy
import logging
import platform
import sys
import time
from collections import defaultdict
from collections import deque

import evdev
import gym
import numpy as np
import pygame
from easydict import EasyDict
from evdev import ecodes, InputDevice

from pvp.experiments.carla.di_drive.core.envs.simple_carla_env import SimpleCarlaEnv
from pvp.experiments.carla.di_drive.demo.simple_rl.env_wrapper import ContinuousBenchmarkEnvWrapper
from pvp.experiments.carla.di_drive.demo.simple_rl.sac_train import compile_config
from pvp.utils.print_dict_utils import merge_dicts
from pvp.utils.utils import ForceFPS, merge_dicts

# Someone change the logging config. So we have to revert them here.
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger = logging.getLogger(__file__)

is_windows = "Win" in platform.system()


def safe_clip(array, min_val, max_val):
    array = np.nan_to_num(array.astype(np.float64), copy=False, nan=0.0, posinf=max_val, neginf=min_val)
    return np.clip(array, min_val, max_val).astype(np.float64)


train_config = dict(
    obs_mode="birdview",
    force_fps=0,
    disable_vis=False,
    port=9000,
    enable_takeover=True,
    show_text=True,
    normalize_obs=False,
    disable_brake=True,
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
            # obs = ..., We will autofill this in HumanInTheLoopCARLAEnv
        ),
        col_is_failure=True,
        stuck_is_failure=False,
        wrong_direction_is_failure=False,
        off_route_is_failure=True,
        off_road_is_failure=True,
        ignore_light=True,
        off_route_distance=10.0,
        visualize=dict(type="vis", outputs=['show'], location="center"),
        # visualize=None,
        manager=dict(collect=dict(
            auto_reset=True,
            shared_memory=False,
            context='spawn',
            max_retry=1,
        ), eval=dict()),
        wrapper=dict(
            # Collect and eval suites for training
            collect=dict(suite='NoCrashTown01-v1'),
        ),
    ),
)


class SteeringWheelController:
    RIGHT_SHIFT_PADDLE = 4
    LEFT_SHIFT_PADDLE = 5
    STEERING_MAKEUP = 1.5

    def __init__(self, disable=False):
        self.disable = disable
        if not self.disable:
            pygame.display.init()
            pygame.joystick.init()
            assert pygame.joystick.get_count() > 0, "Please connect the steering wheel!"
            print("Successfully Connect your Steering Wheel!")

            ffb_device = evdev.list_devices()[0]
            self.ffb_dev = InputDevice(ffb_device)

            self.joystick = pygame.joystick.Joystick(0)

        self.right_shift_paddle = False
        self.left_shift_paddle = False

        self.button_circle = False
        self.button_rectangle = False
        self.button_triangle = False
        self.button_x = False

        self.button_up = False
        self.button_down = False
        self.button_right = False
        self.button_left = False

    def process_input(self, speed_kmh):
        if self.disable:
            return [0.0, 0.0]

        if not self.joystick.get_init():
            self.joystick.init()

        pygame.event.pump()

        if is_windows:
            raise ValueError("We have not yet tested windows.")
            steering = (-self.joystick.get_axis(0)) / 1.5
            throttle = (1 - self.joystick.get_axis(1)) / 2
            brake = (1 - self.joystick.get_axis(3)) / 2
        else:
            # print("Num axes: ", self.joystick.get_numaxes())

            # Our wheel can provide values in [-1.5, 1.5].
            steering = (-self.joystick.get_axis(0)) / 1.5  # 0th axis is the wheel

            # 2nd axis is the right paddle. Range from 0 to 1
            # 3rd axis is the middle paddle. Range from 0 to 1
            # Of course then 1st axis is the left paddle.

            # print("Raw throttle: {}, raw brake: {}".format(self.joystick.get_axis(2), self.joystick.get_axis(3)))
            raw_throttle = self.joystick.get_axis(2)
            raw_brake = self.joystick.get_axis(3)
            # It is possible that the paddles always return 0 (should be 1 if not pressed) after initialization.
            if abs(raw_throttle) < 1e-6:
                raw_throttle = 1.0 - 1e-6
            if abs(raw_brake) < 1e-6:
                raw_brake = 1.0 - 1e-6
            throttle = (1 - raw_throttle) / 2
            brake = (1 - raw_brake) / 2

        self.right_shift_paddle = True if self.joystick.get_button(self.RIGHT_SHIFT_PADDLE) else False
        self.left_shift_paddle = True if self.joystick.get_button(self.LEFT_SHIFT_PADDLE) else False

        # self.print_debug_message()

        self.button_circle = True if self.joystick.get_button(2) else False
        self.button_rectangle = True if self.joystick.get_button(1) else False
        self.button_triangle = True if self.joystick.get_button(3) else False
        self.button_x = True if self.joystick.get_button(0) else False

        if self.button_x:
            logger.warning("X is pressed. Exit ...")
            raise KeyboardInterrupt()

        self.maybe_pause()

        hat = self.joystick.get_hat(0)
        self.button_up = True if hat[-1] == 1 else False
        self.button_down = True if hat[-1] == -1 else False
        self.button_left = True if hat[0] == -1 else False
        self.button_right = True if hat[0] == 1 else False

        self.feedback(speed_kmh)

        return [-steering * self.STEERING_MAKEUP, (throttle - brake)]

    def maybe_pause(self):
        paused = False
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN and event.button == 3:  # Triangle button pressed
                paused = not paused  # Toggle pause
                # Wait for button release
                while True:
                    event_happened = False
                    for event in pygame.event.get():
                        if event.type == pygame.JOYBUTTONUP and event.button == 3:
                            event_happened = True
                            break
                    if event_happened:
                        break
                    pygame.time.delay(100)

                # Wait for the next button press to unpause
                while True:
                    event_happened = False
                    for event in pygame.event.get():
                        if event.type == pygame.JOYBUTTONDOWN and event.button == 3:
                            event_happened = True
                            break
                    if event_happened:
                        break
                    pygame.time.delay(100)

                # Button pressed again, unpause
                paused = False

                # Wait for button release before exiting
                while True:
                    event_happened = False
                    for event in pygame.event.get():
                        if event.type == pygame.JOYBUTTONUP and event.button == 3:
                            event_happened = True
                            break
                    if event_happened:
                        break
                    pygame.time.delay(100)

    def reset(self):
        if self.disable:
            self.right_shift_paddle = False
            self.left_shift_paddle = False
            return

        self.right_shift_paddle = False
        self.left_shift_paddle = False
        self.button_circle = False
        self.button_rectangle = False
        self.button_triangle = False
        self.button_x = False
        self.button_up = False
        self.button_down = False
        self.button_right = False
        self.button_left = False
        self.joystick.quit()
        pygame.event.clear()

        val = int(65535)
        self.ffb_dev.write(ecodes.EV_FF, ecodes.FF_AUTOCENTER, val)

    def feedback(self, speed_kmh):
        assert not self.disable
        offset = 5000
        total = 50000
        val = int(total * min(speed_kmh / 80, 1) + offset)
        self.ffb_dev.write(ecodes.EV_FF, ecodes.FF_AUTOCENTER, val)

    def print_debug_message(self):
        msg = "Left: {}, Right: {}, Event: ".format(
            self.joystick.get_button(self.LEFT_SHIFT_PADDLE), self.joystick.get_button(self.RIGHT_SHIFT_PADDLE)
        )
        for e in pygame.event.get():
            msg += str(e.type)
        print(msg)


class HumanInTheLoopCARLAEnv(ContinuousBenchmarkEnvWrapper):
    def __init__(
        self,
        external_config=None,
    ):
        config = copy.deepcopy(train_config)

        # If disable visualization, we don't need to create the main camera (third person view)
        disable_vis = external_config.get("disable_vis", None)
        if disable_vis is None:
            disable_vis = config["disable_vis"]
        if disable_vis:
            sensors = []
        else:
            sensors = [
                dict(
                    name='vis',
                    type='rgb',
                    size=[2100, 1400],
                    position=[-5.5, 0, 2.8],
                    rotation=[-15, 0, 0],
                )
            ]

        # Add camera to generate agent observation
        if external_config["obs_mode"] == "birdview":
            sensors.append(
                dict(
                    name="birdview",
                    type='bev',
                    size=[84, 84],
                    pixels_per_meter=2,
                    pixels_ahead_vehicle=16,
                )
            )
        elif external_config["obs_mode"] in ["first", "firststack"]:
            sensors.append(
                dict(
                    name='rgb',
                    type='rgb',
                    size=[320, 180],

                    # PZH Note: To recap, the default config contains:
                    # 'position':[2.0, 0.0, 1.4],
                    # 'rotation':[0, 0, 0],
                )
            )
            if external_config["obs_mode"] == "firststack":
                self._frame_stack = deque(maxlen=5)
        elif external_config["obs_mode"] == "birdview42":
            sensors.append(dict(name="birdview", type='bev', size=[42, 42], pixels_per_meter=6))
        else:
            raise ValueError("Unknown obs mode: {}".format(external_config["obs_mode"]))

        # Tweak the config
        config["env"]["simulator"]["obs"] = tuple(sensors)
        config["env"]["obs_mode"] = external_config["obs_mode"]
        config = EasyDict(config)
        if external_config is not None and external_config:
            config = EasyDict(merge_dicts(config, external_config))
        if config["show_text"] is False:
            config["env"]["visualize"]["show_text"] = False
        if config["disable_vis"]:
            config["env"]["visualize"] = None
        self.monitor_index = config["env"].get("visualize", {}).get("monitor_index", 0)
        compiled_config = compile_config(config)
        port = compiled_config["port"]
        self.main_config = compiled_config
        super(HumanInTheLoopCARLAEnv, self).__init__(
            SimpleCarlaEnv(compiled_config.env, "localhost", port, None), compiled_config.env.wrapper.collect
        )

        # Set up controller
        if compiled_config["enable_takeover"]:
            self.controller = SteeringWheelController(disable=False)
        else:
            self.controller = None

        # Set up some variables
        self.last_takeover = False
        self.episode_recorder = defaultdict(list)
        force_fps = self.main_config["force_fps"]
        self.force_fps = ForceFPS(fps=force_fps, start=(force_fps > 0))
        self.normalize_obs = self.main_config["normalize_obs"]

    def step(self, action):

        # Get control signal from human
        if self.controller is not None:
            try:
                human_action = self.controller.process_input(self.env._simulator_databuffer['state']['speed_kmh'])
            except KeyboardInterrupt as e:
                self.close()
                raise e
            if self.main_config["disable_brake"] and human_action[1] < 0.0:
                human_action[1] = 0.0
            takeover = self.controller.left_shift_paddle or self.controller.right_shift_paddle
        else:
            human_action = [0, 0]
            takeover = False

        # Step the environment
        o, r, d, info = super(HumanInTheLoopCARLAEnv, self).step(human_action if takeover else action)

        # Postprocess the environment return
        self.episode_recorder["reward"].append(r)
        info["takeover_start"] = True if not self.last_takeover and takeover else False
        info["takeover"] = takeover and not info["takeover_start"]
        condition = info["takeover_start"]
        if not condition:
            self.episode_recorder["cost"].append(0)
            info["takeover_cost"] = 0
        else:
            cost = self.get_takeover_cost(human_action, action)
            self.episode_recorder["cost"].append(cost)
            info["takeover_cost"] = cost
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

        # Render
        if not self.main_config["disable_vis"]:
            self.render()
        self.force_fps.sleep_if_needed()

        return o, float(r[0]), d, info

    def native_cost(self, info):
        if info["off_route"] or info["off_road"] or info["collided"] or info["wrong_direction"]:
            return 1
        else:
            return 0

    def get_takeover_cost(self, human_action, agent_action):
        takeover_action = safe_clip(np.array(human_action), -1, 1)
        agent_action = safe_clip(np.array(agent_action), -1, 1)
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
        obs = super(HumanInTheLoopCARLAEnv, self).reset()
        self.force_fps.clear()
        return obs

    def postprocess_obs(self, raw_obs, in_reset=False):
        speed = float(raw_obs["speed_kmh"])  # in km/h
        normalized_speed = min(speed, self.env._max_speed_kmh) / self.env._max_speed_kmh
        if self.main_config["obs_mode"] in ["birdview", "birdview42"]:
            obs = raw_obs["birdview"][..., [0, 1, 5, 6, 8]]
            if self.normalize_obs:
                obs = obs.astype(np.float32) / 255
            obs_out = {
                "image": obs,
                'speed': np.array([normalized_speed]),
            }
        elif self.main_config["obs_mode"] == "first":
            obs = raw_obs["rgb"]
            if self.normalize_obs:
                obs = obs.astype(np.float32) / 255
            obs_out = {
                'image': obs,
                'speed': np.array([normalized_speed]),
            }
        elif self.main_config["obs_mode"] == "firststack":
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
        return gym.spaces.Box(-1.0, 1.0, shape=(2, ))

    @property
    def observation_space(self):
        if self.main_config["obs_mode"] == "birdview":
            return gym.spaces.Dict(
                {
                    "image": gym.spaces.Box(low=0, high=255, shape=(84, 84, 5), dtype=np.uint8),
                    "speed": gym.spaces.Box(0., 1.0, shape=(1, ))
                }
            )
        elif self.main_config["obs_mode"] == "first":
            return gym.spaces.Dict(
                {
                    "image": gym.spaces.Box(low=0, high=255, shape=(180, 320, 3), dtype=np.uint8),
                    "speed": gym.spaces.Box(0., 1.0, shape=(1, ))
                }
            )
        elif self.main_config["obs_mode"] == "firststack":
            return gym.spaces.Box(low=0, high=255, shape=(180, 320, 5), dtype=np.uint8)

        elif self.main_config["obs_mode"] == "birdview42":
            return gym.spaces.Dict(
                {
                    "image": gym.spaces.Box(low=0, high=255, shape=(42, 42, 5), dtype=np.uint8),
                    "speed": gym.spaces.Box(0., 1.0, shape=(1, ))
                }
            )
        else:
            raise ValueError("Wrong obs_mode")

    def render(self):
        return super(HumanInTheLoopCARLAEnv, self).render(takeover=self.last_takeover, monitor_index=self.monitor_index)


if __name__ == "__main__":
    env = HumanInTheLoopCARLAEnv({
        "obs_mode": "firststack",
        "force_fps": 10,
        "enable_takeover": True,
    })
    env.seed(0)
    o = env.reset()
    st = time.time()
    count = 0
    while True:
        o, r, d, i = env.step([0., 1.0])
        count += 1
        ct = time.time()
        fps = 1 / (ct - st)
        st = ct
        if d:
            env.reset()
