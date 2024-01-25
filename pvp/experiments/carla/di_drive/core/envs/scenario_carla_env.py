import os
import time
from datetime import datetime
from typing import Any, Dict

import numpy as np
from gym import spaces

from pvp.experiments.carla.di_drive.core.simulators import CarlaScenarioSimulator
from pvp.experiments.carla.di_drive.core.utils.others.visualizer import Visualizer
from pvp.experiments.carla.di_drive.core.utils.simulator_utils.carla_utils import visualize_birdview
from .base_carla_env import BaseCarlaEnv


class ScenarioCarlaEnv(BaseCarlaEnv):
    """
    Carla Scenario Environment with a single hero vehicle. It uses ``CarlaScenarioSimulator`` to load scenario
    configurations and interacts with Carla server to get running status. The Env is initialized with a scenario
    config, which could be a route with scenarios or a single scenario. The observation, sensor settings and visualizer
    are the same with `SimpleCarlaEnv`. The reward is derived based on the scenario criteria in each tick. The criteria
    is also related to success and failure judgement which is used to end an episode.

    When created, it will initialize environment with config and Carla TCP host & port. This method will NOT create
    the simulator instance. It only creates some data structures to store information when running env.

    :Arguments:
        - cfg (Dict): Env config dict.
        - host (str, optional): Carla server IP host. Defaults to 'localhost'.
        - port (int, optional): Carla server IP port. Defaults to 9000.
        - tm_port (Optional[int], optional): Carla Traffic Manager port. Defaults to None.

    :Interfaces: reset, step, close, is_success, is_failure, render, seed

    :Properties:
        - hero_player (carla.Actor): Hero vehicle in simulator.
    """

    action_space = spaces.Dict({})
    observation_space = spaces.Dict({})
    config = dict(
        simulator=dict(),
        finish_reward=100,
        visualize=None,
        outputs=[],
        output_dir='',

        # FIXME(xxx): Check these configs. I am not sure they are correct here!!!
        col_is_failure=False,
        stuck_is_failure=False,
        ignore_light=False,
        ran_light_is_failure=False,
        off_road_is_failure=False,
        wrong_direction_is_failure=False,
        off_route_is_failure=False,
        off_route_distance=6,
        success_distance=5,
        success_reward=10,
        stuck_len=300,
        max_speed=5,
    )

    def __init__(
        self,
        cfg: Dict,
        host: str = 'localhost',
        port: int = None,
        tm_port: int = None,
        **kwargs,
    ) -> None:
        """
        Initialize environment with config and Carla TCP host & port.
        """
        super().__init__(cfg, **kwargs)
        self.cfg = cfg

        self._simulator_cfg = self._cfg.simulator
        self._carla_host = host
        self._carla_port = port
        self._carla_tm_port = tm_port

        self._use_local_carla = False
        if self._carla_host != 'localhost':
            self._use_local_carla = True
        self._simulator = None

        self._output_dir = self._cfg.output_dir
        self._outputs = self._cfg.outputs

        self._finish_reward = self._cfg.finish_reward
        self._is_success = False
        self._is_failure = False
        self._collided = False

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
        self._max_speed = self._cfg.max_speed
        self._collided = False
        self._stuck = False
        self._ran_light = False
        self._off_road = False
        self._wrong_direction = False
        self._off_route = False
        self._reward = 0
        self._episode_reward = 0
        self._last_steer = 0
        self._last_distance = None

        from pvp.experiments.carla.di_drive.core.utils.env_utils.stuck_detector import StuckDetector
        self._stuck_detector = StuckDetector(self._cfg.stuck_len)

        self._tick = 0
        self._timeout = float('inf')
        self._launched_simulator = False
        self._config = None

        self._visualize_cfg = self._cfg.visualize
        self._simulator_databuffer = dict()
        self._visualizer = None

    def _init_carla_simulator(self) -> None:
        if not self._use_local_carla:
            print("------ Run Carla on Port: %d, GPU: %d ------" % (self._carla_port, 0))
            # self.carla_process = subprocess.Popen()
            self._simulator = CarlaScenarioSimulator(
                cfg=self._simulator_cfg,
                client=None,
                host=self._carla_host,
                port=self._carla_port,
                tm_port=self._carla_tm_port,

                # xxx: We set the timeout here to 60s.
                timeout=60.0
            )
        else:
            print('------ Using Remote Carla @ {}:{} ------'.format(self._carla_host, self._carla_port))
            self._simulator = CarlaScenarioSimulator(
                cfg=self._simulator_cfg,
                client=None,
                host=self._carla_host,
                port=self._carla_port,
                tm_port=self._carla_tm_port,

                # xxx: We set the timeout here to 60s.
                timeout=60
            )
        self._launched_simulator = True

    def reset(self, config: Any) -> Dict:
        """
        Reset environment to start a new episode, with provided reset params. If there is no simulator, this method will
        create a new simulator instance. The reset param is sent to simulator's ``init`` method to reset simulator,
        then reset all statues recording running states, and create a visualizer if needed. It returns the first frame
        observation.

        :Arguments:
            - config (Any): Configuration instance of the scenario

        :Returns:
            Dict: The initial observation.
        """
        if not self._launched_simulator:
            self._init_carla_simulator()
        self._config = config

        self._simulator.init(self._config)

        if self._visualize_cfg is not None:
            if self._visualizer is not None:
                self._visualizer.done()
            else:
                self._visualizer = Visualizer(self._visualize_cfg)

            if 'Route' in config.name:
                config_name = os.path.splitext(os.path.split(config.scenario_file)[-1])[0]
            else:
                config_name = config.name
            vis_name = "{}_{}".format(config_name, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

            self._visualizer.init(vis_name)

        self._simulator_databuffer.clear()
        self._collided = False
        self._criteria_last_value = dict()
        self._is_success = False
        self._is_failure = False

        # if 'col_is_failure' in kwargs:
        #     self._col_is_failure = kwargs['col_is_failure']
        # if 'stuck_is_failure' in kwargs:
        #     self._stuck_is_failure = kwargs['stuck_is_failure']
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

    def step(self, action):
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
            self._simulator_databuffer.update({'action': action})
        self._simulator.run_step()
        self._tick += 1

        obs = self.get_observations()
        self._collided = self._simulator.collided
        self._stuck = self._stuck_detector.stuck
        self._ran_light = self._simulator.ran_light
        self._off_road = self._simulator.off_road
        self._wrong_direction = self._simulator.wrong_direction

        res = self._simulator.scenario_manager.get_scenario_status()
        if res == 'SUCCESS':
            self._is_success = True
        elif res in ['FAILURE', 'INVALID']:
            self._is_failure = True

        self._reward, reward_info = self.compute_reward()
        self._episode_reward += self._reward

        done = self.is_success() or self.is_failure()
        if done:
            self._simulator.end_scenario()
            self._conclude_scenario(self._config)
            if self._visualizer is not None:
                self._visualizer.done()

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
                'failure': self.is_failure(),
                'success': self.is_success(),
            }
        )

        location = self._simulator_databuffer['state']['location'][:2]
        target = self._simulator_databuffer['navigation']['target']
        self._off_route = bool(np.linalg.norm(location - target) >= self._off_route_distance)

        info.update(reward_info)

        return obs, self._reward, done, info

    def close(self):
        """
        Delete simulator & visualizer instances and close environment.
        """
        if self._launched_simulator:
            self._simulator.clean_up()
            self._simulator._set_sync_mode(False)
            del self._simulator
            self._launched_simulator = False
        if self._visualizer is not None:
            self._visualizer.done()

    def is_success(self) -> bool:
        """
        Check if the task succeed. It only happens when behavior tree ends successfully.

        :Returns:
            bool: Whether success.
        """
        res = self._simulator.scenario_manager.get_scenario_status()
        if res == 'SUCCESS':
            return True
        return False

    def is_failure(self) -> bool:
        """
        Check if the task fails. It may happen when behavior tree ends unsuccessfully or some criteria trigger.

        :Returns:
            bool: Whether failure.
        """
        res = self._simulator.scenario_manager.get_scenario_status()
        if res in ['FAILURE', 'INVALID']:
            return True

        # xxx: We merge the termination conditions from the simple_carla_env
        if self._stuck_is_failure and self._stuck:
            print("The episode is terminated because of stuck.")
            return True
        if self._col_is_failure and self._collided:
            print("The episode is terminated because of collision.")
            return True
        if self._ran_light_is_failure and self._ran_light:
            print("The episode is terminated because of running into wrong lights.")
            return True
        if self._off_road_is_failure and self._off_road:
            print("The episode is terminated because of driving out of road.")
            return True
        if self._wrong_direction_is_failure and self._wrong_direction:
            print("The episode is terminated because of driving in wrong direction.")
            return True
        if self._off_route_is_failure and self._off_route:
            print("The episode is terminated because of driving out of route.")
            return True
        # TODO(xxx): We hardcode the system frwequency here!
        if self._tick / 10 > self._timeout:
            print(
                "The episode is terminated because of timeout. (Timeout: {}, current tick: {})".format(
                    self._timeout, self._tick
                )
            )
            return True

        return False

    def get_observations(self):
        """
        Get observations from simulator. The sensor data, navigation, state and information in simulator
        are used, while not all these are added into observation dict.

        :Returns:
            Dict: Observation dict.
        """
        obs = dict()
        state = self._simulator.get_state()
        sensor_data = self._simulator.get_sensor_data()
        navigation = self._simulator.get_navigation()
        information = self._simulator.get_information()

        self._simulator_databuffer['state'] = state
        self._simulator_databuffer['navigation'] = navigation
        self._simulator_databuffer['information'] = information

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
                'speed': np.float32(state['speed']),
                'speed_limit': np.float32(navigation['speed_limit']),
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
                raise ValueError("visualize type {} not in sensor data!".format(self._visualize_cfg.type))
            self._render_buffer = sensor_data[self._visualize_cfg.type].copy()
            if self._visualize_cfg.type == 'birdview':
                self._render_buffer = visualize_birdview(self._render_buffer)
        return obs

    def compute_reward(self):
        """
        Compute reward for current frame, and return details in a dict. In short, it contains goal reward,
        route following reward calculated by criteria in current and last frame, and failure reward by checking criteria
        in each frame.

        :Returns:
            Tuple[float, Dict]: Total reward value and detail for each value.
        """
        goal_reward = 0
        if self._is_success:
            goal_reward += self._finish_reward

        elif self._is_failure:
            goal_reward -= self._finish_reward

        criteria_dict = self._simulator.get_criteria()

        # print("Criteria: ", criteria_dict)

        failure_reward = 0
        complete_reward = 0
        for info, value in criteria_dict.items():
            if value[0] == 'FAILURE':
                if info in self._criteria_last_value and value[1] != self._criteria_last_value[info][1]:
                    if 'Collision' in info:
                        failure_reward -= 10
                    elif 'RunningRedLight' in info:
                        failure_reward -= 10
                    elif 'OutsideRouteLanes' in info:
                        failure_reward -= 10
            if 'RouteCompletion' in info and info in self._criteria_last_value:
                complete_reward = self._finish_reward / 100 * (value[1] - self._criteria_last_value[info][1])
            self._criteria_last_value[info] = value

        reward_info = dict()
        reward_info['goal_reward'] = goal_reward
        reward_info['complete_reward'] = complete_reward
        reward_info['failure_reward'] = failure_reward
        reward_info["route_completion"] = self._criteria_last_value["RouteCompletionTest"][1] / 100

        total_reward = goal_reward + failure_reward + complete_reward

        return total_reward, reward_info

    def render(self):
        """
        Render a runtime visualization on screen, save a gif video file according to visualizer config.
        The main canvas is from a specific sensor data. It only works when 'visualize' is set in config dict.
        """
        if self._visualizer is None:
            return

        render_info = {
            'collided': self._collided,
            'reward': self._reward,
            'episode_reward': self._episode_reward,
            'tick': self._tick,
            'end_timeout': self._simulator.end_timeout,
            'distance_to_go': self._simulator.distance_to_go,
            'total_distance': self._simulator.total_diatance,
            "FPS": self.get_fps(),
        }
        render_info.update(self._simulator_databuffer['state'])
        render_info.update(self._simulator_databuffer['navigation'])
        render_info.update(self._simulator_databuffer['information'])
        render_info.update(self._simulator_databuffer['action'])

        self._visualizer.paint(self._render_buffer, render_info)
        self._visualizer.run_visualize()

    def seed(self, seed: int) -> None:
        """
        Set random seed for environment.

        :Arguments:
            - seed (int): Random seed value.
        """
        print('[ENV] Setting seed:', seed)
        np.random.seed(seed)

    def __repr__(self) -> str:
        return "ScenarioCarlaEnv with host %s, port %s." % (self._carla_host, self._carla_port)

    def _conclude_scenario(self, config: Any) -> None:
        """
        Provide feedback about success/failure of a scenario
        """

        # Create the filename
        current_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        junit_filename = None
        filename = None
        config_name = config.name

        if self._output_dir != '':
            os.makedirs(self._output_dir, exist_ok=True)
            config_name = os.path.join(self._output_dir, config_name)
        if 'junit' in self._outputs:
            junit_filename = config_name + '_' + current_time + ".xml"
        if 'file' in self._outputs:
            filename = config_name + '_' + current_time + "txt"

        if self._simulator.scenario_manager.analyze_scenario(True, filename, junit_filename):
            self._is_failure = True
        else:
            self._is_success = True

    @property
    def hero_player(self):
        return self._simulator.hero_player
