import time
from collections import deque
from typing import Any, Dict, Optional, Tuple

import carla
import numpy as np
from gym import spaces

from pvp.experiments.carla.di_drive.core.simulators import CarlaSimulator
from pvp.experiments.carla.di_drive.core.utils.env_utils.stuck_detector import StuckDetector
from pvp.experiments.carla.di_drive.core.utils.others.visualizer import Visualizer
from pvp.experiments.carla.di_drive.core.utils.simulator_utils.carla_utils import visualize_birdview
from .base_carla_env import BaseCarlaEnv


def dist(loc1, loc2):
    return ((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)**0.5


class SimpleCarlaEnv(BaseCarlaEnv):
    """
    A simple deployment of Carla Environment with single hero vehicle. It use ``CarlaSimulator`` to interact with
    Carla server and gets running status. The observation is obtained from simulator's state, information and
    sensor data, along with reward which can be calculated and retrived.

    When created, it will initialize environment with config and Carla TCP host & port. This method will NOT create
    simulator instance. It only creates some data structures to store information when running env.

    :Arguments:
        - cfg (Dict): Env config dict.
        - host (str, optional): Carla server IP host. Defaults to 'localhost'.
        - port (int, optional): Carla server IP port. Defaults to 9000.
        - tm_port (Optional[int], optional): Carla Traffic Manager port. Defaults to None.

    :Interfaces: reset, step, close, is_success, is_failure, render, seed

    :Properties:
        - hero_player (carla.Actor): Hero vehicle in simulator.
    """

    metadata = {'render.modes': ['rgb_array']}
    action_space = spaces.Dict({})
    observation_space = spaces.Dict({})
    config = dict(

        # Reward setting
        success_distance=5.0,
        success_reward=10.0,
        fail_penalty=5.0,

        # System setting
        ignore_light=True,
        max_speed_kmh=40.0,
        stuck_len=300,
        off_route_distance=6.0,
        simulator=dict(),
        visualize=None,

        # Termination
        stuck_is_failure=False,
        ran_light_is_failure=False,
        wrong_direction_is_failure=False,
        off_route_is_failure=False,
        col_is_failure=True,
        off_road_is_failure=True,
    )

    def __init__(
        self,
        cfg: Dict,
        host: str = 'localhost',
        port: int = 9000,
        tm_port: Optional[int] = None,
        carla_timeout: Optional[int] = 60.0,
        **kwargs,
    ) -> None:
        """
        Initialize environment with config and Carla TCP host & port.
        """
        super().__init__(cfg, **kwargs)

        self._simulator_cfg = self._cfg.simulator
        self._carla_host = host
        self._carla_port = port
        self._carla_tm_port = tm_port
        self._carla_timeout = carla_timeout

        self._use_local_carla = False
        if self._carla_host != 'localhost':
            self._use_local_carla = True
        self._simulator = None

        self._col_is_failure = self._cfg.col_is_failure
        self._stuck_is_failure = self._cfg.stuck_is_failure
        self._ignore_light = self._cfg.ignore_light
        self._ran_light_is_failure = self._cfg.ran_light_is_failure
        self._off_road_is_failure = self._cfg.off_road_is_failure
        self._wrong_direction_is_failure = self._cfg.wrong_direction_is_failure
        self._off_route_is_failure = self._cfg.off_route_is_failure
        self._off_route_distance = self._cfg.off_route_distance

        self._success_distance = self._cfg.success_distance
        self._success_reward = self._cfg.success_reward
        self._max_speed_kmh = self._cfg.max_speed_kmh
        self._collided = False
        self._stuck = False
        self._ran_light = False
        self._off_road = False
        self._wrong_direction = False
        self._off_route = False
        self._stuck_detector = StuckDetector(self._cfg.stuck_len)

        self._tick = 0
        self._total_step = 0
        self._total_time = time.time()
        self._takeover_recorder = deque(maxlen=2000)
        self._timeout = float('inf')
        self._launched_simulator = False

        self._visualize_cfg = self._cfg.visualize
        # We can disable the text when displaying the birdview with visualizer.
        # self._visualize_cfg["show_text"] = self._cfg["obs_mode"] != "birdview"

        self._simulator_databuffer = dict()
        self._visualizer = None

    def _init_carla_simulator(self) -> None:
        if not self._use_local_carla:
            print("------ Run Carla on Port: %d, GPU: %d ------" % (self._carla_port, 0))
            # self.carla_process = subprocess.Popen()
            self._simulator = CarlaSimulator(
                cfg=self._simulator_cfg,
                client=None,
                host=self._carla_host,
                port=self._carla_port,
                tm_port=self._carla_tm_port,
                timeout=self._carla_timeout,
            )
        else:
            print('------ Using Remote carla @ {}:{} ------'.format(self._carla_host, self._carla_port))
            self._simulator = CarlaSimulator(
                cfg=self._simulator_cfg,
                client=None,
                host=self._carla_host,
                port=self._carla_port,
                tm_port=self._carla_tm_port
            )
        self._launched_simulator = True

    def reset(self, **kwargs) -> Dict:
        """
        Reset environment to start a new episode, with provided reset params. If there is no simulator, this method will
        create a new simulator instance. The reset param is sent to simulator's ``init`` method to reset simulator,
        then reset all statues recording running states, and create a visualizer if needed. It returns the first frame
        observation.

        :Returns:
            Dict: The initial observation.
        """
        if not self._launched_simulator:
            self._init_carla_simulator()

        self._simulator.init(**kwargs)

        if self._visualize_cfg is not None:
            if self._visualizer is not None:
                self._visualizer.done()
            else:
                self._visualizer = Visualizer(self._visualize_cfg)

            if 'name' in kwargs:
                vis_name = kwargs['name']
            else:
                vis_name = "{}_{}".format(
                    self._simulator.town_name, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                )

            self._visualizer.init(vis_name)

        if 'col_is_failure' in kwargs:
            self._col_is_failure = kwargs['col_is_failure']
        if 'stuck_is_failure' in kwargs:
            self._stuck_is_failure = kwargs['stuck_is_failure']
        self._simulator_databuffer.clear()
        self._collided = False
        self._stuck = False
        self._ran_light = False
        self._off_road = False
        self._wrong_direction = False
        self._off_route = False
        self._stuck_detector.clear()
        self._tick = 0
        self._reward = 0
        self._episode_reward = 0
        self._last_steer = 0
        self._last_distance = None
        self._timeout = self._simulator.end_timeout

        return self.get_observations()

    def step(self, action: Dict) -> Tuple[Any, float, bool, Dict]:
        """
        Run one time step of environment, get observation from simulator and calculate reward. The environment will
        be set to 'done' only at success or failure. And if so, all visualizers will end. Its interfaces follow
        the standard definition of ``gym.Env``.

        :Arguments:
            - action (Dict): Action provided by policy.

        :Returns:
            Tuple[Any, float, bool, Dict]: A tuple contains observation, reward, done and information.
        """
        if action is not None:
            self._simulator.apply_control(action)
            self._simulator_databuffer['action'] = action
        else:
            self._simulator_databuffer['action'] = dict()
        timestamp = self._simulator.run_step()
        self._tick += 1
        self._total_step += 1

        obs = self.get_observations()

        self._collided = self._simulator.collided
        self._stuck = self._stuck_detector.stuck
        self._ran_light = self._simulator.ran_light
        self._off_road = self._simulator.off_road
        self._wrong_direction = self._simulator.wrong_direction

        location = self._simulator_databuffer['state']['location'][:2]
        target = self._simulator_databuffer['navigation']['target']

        # print("Distance to target: ", np.linalg.norm(location - target), self._off_route_distance)
        self._off_route = float(np.linalg.norm(location - target) >= self._off_route_distance)

        self._reward, reward_info = self.compute_reward()
        self._episode_reward += self._reward

        info = self._simulator.get_information()
        info.update(reward_info)
        info.update(
            {
                'collided': self._collided,
                'stuck': self._stuck,
                'ran_light': self._ran_light,
                'off_road': self._off_road,
                'wrong_direction': self._wrong_direction,
                'off_route': self._off_route,
                'timeout': self._tick > self._timeout,
                'success': self.is_success(),
                "route_completion": 1 - self._simulator.distance_to_go / self._simulator.total_diatance,
                "distance_total": self._simulator.total_diatance,
                "distance_current": self._simulator.total_diatance - self._simulator.distance_to_go,
                "distance_to_go": self._simulator.distance_to_go
            }
        )

        # print({"route_completion": 1 - self._simulator.distance_to_go / self._simulator.total_diatance,
        #        "total_distance": self._simulator.total_diatance,
        #        "current_distance": self._simulator.distance_to_go})

        done = self.is_success() or self.is_failure()
        if done:
            self._simulator.clean_up()
            if self._visualizer is not None:
                self._visualizer.done()
                self._visualizer = None

        return obs, float(self._reward), done, info

    def close(self) -> None:
        """
        Delete simulator & visualizer instances and close the environment.
        """
        if self._launched_simulator:
            self._simulator.clean_up()
            self._simulator._set_sync_mode(False)
            del self._simulator
            self._launched_simulator = False
        if self._visualizer is not None:
            self._visualizer.done()
            self._visualizer = None

    def is_success(self) -> bool:
        """
        Check if the task succeed. It only happens when hero vehicle is close to target waypoint.

        :Returns:
            bool: Whether success.
        """
        if self._simulator.distance_to_go < self._success_distance:
            print(
                "Yes! Current distance_to_go {} and success_distance {}, so you success!".format(
                    self._simulator.distance_to_go, self._success_distance
                )
            )
            return True
        return False

    def is_failure(self) -> bool:
        """
        Check if env fails. olliding, being stuck, running light, running off road, running in
        wrong direction according to config. It will certainly happen when time is out.

        :Returns:
            bool: Whether failure.
        """
        if self._stuck_is_failure and self._stuck:
            raise ValueError("xxx: We don't suggest to turn on stuck termination condiction!")
            return True
        if self._col_is_failure and self._collided:
            return True
        if self._ran_light_is_failure and self._ran_light:
            return True
        if self._off_road_is_failure and self._off_road:
            return True
        if self._wrong_direction_is_failure and self._wrong_direction:
            return True
        if self._off_route_is_failure and self._off_route:
            return True
        if self._tick > self._timeout:
            print("What is the timeout?", self._tick, self._timeout)
            return True

        return False

    def get_observations(self) -> Dict:
        """
        Get observations from simulator. The sensor data, navigation, state and information in simulator
        are used, while not all these are added into observation dict.

        :Returns:
            Dict: Observation dict.
        """
        obs = dict()
        state = self._simulator.get_state()  # speed in m/s
        navigation = self._simulator.get_navigation()  # speed_limit in m/s
        sensor_data = self._simulator.get_sensor_data()
        information = self._simulator.get_information()

        state["speed_kmh"] = state["speed"] * 3.6  # in km/h
        navigation["speed_limit_kmh"] = navigation["speed_limit"] * 3.6  # in km/h

        self._simulator_databuffer['state'] = state
        self._simulator_databuffer['navigation'] = navigation
        self._simulator_databuffer['information'] = information
        if 'action' not in self._simulator_databuffer:
            self._simulator_databuffer['action'] = dict()
        if not navigation['agent_state'] == 4 or self._ignore_light:
            self._stuck_detector.tick(state['speed'])

        obs.update(sensor_data)
        obs.update(
            {
                'tick': information['tick'],
                'timestamp': np.float32(information['timestamp']),
                'agent_state': navigation['agent_state'],
                'node': navigation['node'],
                'node_forward': navigation['node_forward'],
                'target': np.float32(navigation['target']),
                'target_forward': np.float32(navigation['target_forward']),
                'command': navigation['command'],
                'speed': np.float32(state['speed']),  # in m/s
                'speed_limit': np.float32(navigation['speed_limit']),  # in m/s
                'speed_kmh': np.float32(state['speed']) * 3.6,  # in m/s
                'speed_limit_kmh': np.float32(navigation['speed_limit']) * 3.6,  # in m/s
                'location': np.float32(state['location']),
                'forward_vector': np.float32(state['forward_vector']),
                'acceleration': np.float32(state['acceleration']),
                'velocity': np.float32(state['velocity']),
                'angular_velocity': np.float32(state['angular_velocity']),
                'rotation': np.float32(state['rotation']),
                'is_junction': np.float32(state['is_junction']),
                'tl_state': state['tl_state'],
                'tl_dis': np.float32(state['tl_dis']),
                'waypoint_list': navigation['waypoint_list'],
                'direction_list': navigation['direction_list'],
            }
        )

        if self._visualizer is not None:
            if self._visualize_cfg.type not in sensor_data:
                raise ValueError(
                    "visualize type {} not in sensor data! {}".format(self._visualize_cfg.type, sensor_data.keys())
                )
            self._render_buffer = sensor_data[self._visualize_cfg.type].copy()
            if self._visualize_cfg.type == 'birdview':
                self._render_buffer = visualize_birdview(self._render_buffer)
        return obs

    def compute_reward(self) -> Tuple[float, Dict]:
        """
        Compute reward for current frame, with details returned in a dict. In short, in contains goal reward,
        route following reward calculated by route length in current and last frame, some navigation attitude reward
        with respective to target waypoint, and failure reward by checking each failure event.

        :Returns:
            Tuple[float, Dict]: Total reward value and detail for each value.
        """

        distance_to_go = self._simulator.distance_to_go
        location = self._simulator_databuffer['state']['location']

        # xxx: Compute a lateral factor, it should be 1 if agent is in the center of lane. We temporarily ignore it
        # for simplicity of the reward function
        lateral_factor = 1
        # from pvp.experiments.carla.di_drive.core.utils.simulator_utils.carla_utils import lane_mid_distance
        # waypoint_list = self._simulator_databuffer['navigation']['waypoint_list']
        # lane_mid_dis = lane_mid_distance(waypoint_list, location)
        # lane_reward = max(0, 1 - lane_mid_dis)
        # lane_reward = 0.1 * lane_reward

        # xxx: Reverse the reward if the agent is in the wrong road / direction
        if self._wrong_direction or self._off_road or self._off_route:
            correct_direction_factor = -1
        else:
            correct_direction_factor = 1

        # xxx: Speed reward, weighted by direction.
        speed = self._simulator_databuffer['state']['speed_kmh']
        speed_reward = 0.1 * (min(speed, self._max_speed_kmh) / self._max_speed_kmh) * correct_direction_factor

        # xxx: Driving reward, the movement toward distance, weighted by direction and lateral distance to lane center.
        target = self._simulator_databuffer['navigation']['target']
        target_distance = dist(target, location)
        cur_distance = distance_to_go + target_distance
        if self._last_distance is None:
            driving_reward = 0
        else:
            driving_reward = (self._last_distance - cur_distance) * correct_direction_factor * lateral_factor
        self._last_distance = cur_distance

        total_reward = driving_reward + speed_reward

        # xxx: Termination reward
        termination_reward = 0
        if self.is_success():
            termination_reward = total_reward = self._success_reward
        elif ((self._col_is_failure and self._collided) or (self._stuck_is_failure and self._stuck)
              or (self._off_road_is_failure and self._off_road)
              or (self._ran_light_is_failure and ((not self._ignore_light) and self._ran_light))
              or (self._wrong_direction_is_failure and self._wrong_direction)
              or (self._off_route_is_failure and self._off_route)):
            termination_reward = total_reward = -self._cfg.fail_penalty

        reward_info = {
            'termination_reward': float(termination_reward),
            'driving_reward': float(driving_reward),
            'speed_reward': float(speed_reward),
            'total_reward': float(total_reward)
        }

        return float(total_reward), reward_info

    def render(self, mode='rgb_array', takeover=None, monitor_index=0) -> None:
        """
        Render a runtime visualization on screen, save a gif or video according to visualizer config.
        The main canvas is from a specific sensor data. It only works when 'visualize' is set in config dict.
        """
        if self._visualizer is None:
            return

        if takeover:
            self._takeover_recorder.append(1)
        else:
            self._takeover_recorder.append(0)

        render_info = {
            'collided': self._collided,
            'off_road': self._off_road,
            'wrong_direction': self._wrong_direction,
            'off_route': self._off_route,
            'reward': self._reward,
            'episode_reward': self._episode_reward,
            'tick': self._tick,
            'end_timeout': self._simulator.end_timeout,
            'distance_to_go': self._simulator.distance_to_go,
            'distance_total': self._simulator.total_diatance,
            "FPS": self.get_fps(),
            'total_step': self._total_step,
            "total_time": time.time() - self._total_time,
            "takeover_rate": np.mean(self._takeover_recorder)
        }
        if takeover is not None:
            render_info["takeover"] = takeover
        render_info.update(self._simulator_databuffer['state'])
        render_info.update(self._simulator_databuffer['navigation'])
        render_info.update(self._simulator_databuffer['information'])
        render_info.update(self._simulator_databuffer['action'])

        self._visualizer.paint(self._render_buffer, render_info, monitor_index=monitor_index)
        self._visualizer.run_visualize()
        return self._visualizer.canvas

    def seed(self, seed: int) -> None:
        """
        Set random seed for environment.

        :Arguments:
            - seed (int): Random seed value.
        """
        print('[ENV] Setting seed:', seed)
        np.random.seed(seed)

    def __repr__(self) -> str:
        return "SimpleCarlaEnv - host %s : port %s" % (self._carla_host, self._carla_port)

    @property
    def hero_player(self) -> carla.Actor:
        return self._simulator.hero_player
