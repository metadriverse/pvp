import logging
import os
import pickle

import gym
import numpy as np

logger = logging.getLogger(__name__)


class SharedControlMonitor(gym.Wrapper):
    """
    Store shared control data from multiple episodes.
    """
    def __init__(self, env: gym.Env, folder: str = 'recorded_data', prefix: str = 'data', save_freq: int = 1000):
        super(SharedControlMonitor, self).__init__(env)
        self.data = {
            'observation': [],
            'action_agent': [],  # The action from external policy, typically from the learning agent (novice).
            'action_behavior': [],  # The action applied to the environment, from the behavior policy.
            'reward': [],
            'cost': [],
            'terminated': [],
            'truncated': [],
            'intervention': [],
            'info': [],
            'episode_count': [],
            'step_count': [],
        }
        self.step_count = 0
        self.last_save_step = 0
        self.save_freq = save_freq
        self.folder = folder
        self.prefix = prefix
        self.episode_count = 0
        self.last_observation = None
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        # TODO: Might need to accommodate gymnasium 5-element tuple return.
        truncated = info.get('TimeLimit.truncated', False)
        self._record_step(self.last_observation, action, reward, done, truncated, info)
        self._check_and_save_data()
        self.last_observation = observation
        if done or truncated:
            self.episode_count += 1
        return observation, reward, done, info

    def reset(self, **kwargs):
        # TODO: Might need to handle the 2-element tuple return in newer gym.
        obs = self.env.reset(**kwargs)
        self.last_observation = obs
        return obs

    def _record_step(self, observation, action, reward, terminated, truncated, info):
        self.data['observation'].append(observation)
        self.data['reward'].append(reward)
        self.data['terminated'].append(terminated)
        self.data['truncated'].append(truncated)
        # self.data['info'].append(info)
        if 'cost' in info:
            self.data['cost'].append(info['cost'])
        self.data['episode_count'].append(self.episode_count)
        self.data['step_count'].append(self.step_count)

        assert 'raw_action' in info, info.keys()
        action_behavior = np.asarray(info['raw_action'])
        assert np.shape(action) == np.shape(action_behavior)
        self.data['action_agent'].append(action)
        self.data['action_behavior'].append(action_behavior)

        self.step_count += 1

    def _check_and_save_data(self):
        if self.step_count - self.last_save_step >= self.save_freq:
            self._save_data(num_save_steps=self.step_count - self.last_save_step)
            self.data = {key: [] for key in self.data}  # Reinitialize data
            self.last_save_step = self.step_count

    def _save_data(self, num_save_steps):
        # Convert lists to numpy arrays before saving
        save_data = {}
        for key in self.data:
            if len(self.data[key]) == 0:
                continue
            data_array = np.array(self.data[key])
            if data_array.shape[-1] == 1:
                data_array = data_array.reshape(-1)
            assert data_array.shape[0] == num_save_steps
            save_data[key] = data_array

        file_name = f"{self.prefix}_step_{self.last_save_step}_{self.step_count}.pkl"
        file_path = os.path.join(self.folder, file_name)
        with open(file_path, 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(
            f"Trajectory data from step {self.last_save_step} to {self.step_count} "
            f"(totally {self.step_count - self.last_save_step} steps) is saved at {file_path}"
        )
        self.data = {key: [] for key in self.data}  # Reinitialize data as empty lists
