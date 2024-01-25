"""This file is temporary useless. But we can use it for evaluation if single-environment frame stacking is required!"""
from typing import Dict, Optional, Union

import numpy as np
from gym import Wrapper, spaces

from pvp.sb3.common.vec_env.base_vec_env import VecEnv
from pvp.sb3.common.vec_env.stacked_observations import StackedDictObservations, StackedObservations


class FrameStack(Wrapper):
    def __init__(self, env: VecEnv, n_stack: int, channels_order: Optional[Union[str, Dict[str, str]]] = None):
        super(FrameStack, self).__init__(env=env)
        self.n_stack = n_stack
        wrapped_obs_space = env.observation_space

        if isinstance(wrapped_obs_space, spaces.Box):
            assert not isinstance(
                channels_order, dict
            ), f"Expected None or string for channels_order but received {channels_order}"
            self.stackedobs = StackedObservations(1, n_stack, wrapped_obs_space, channels_order)

        elif isinstance(wrapped_obs_space, spaces.Dict):
            self.stackedobs = StackedDictObservations(1, n_stack, wrapped_obs_space, channels_order)

        else:
            raise Exception("VecFrameStack only works with gym.spaces.Box and gym.spaces.Dict observation spaces")

        self._observation_space = self.stackedobs.stack_observation_space(wrapped_obs_space)

    def step(self, action):

        o, r, d, i = self.env.step(action)
        o, i = self.stackedobs.update(np.expand_dims(o, 0), [d], [i])
        o = o[0]
        return o, r, d, i

    def reset(self):
        o = self.env.reset()
        o = self.stackedobs.reset(np.expand_dims(o, 0))[0]
        return o


if __name__ == '__main__':
    import gym
    from pvp.sb3.common.atari_wrappers import AtariWrapper

    env_name = "BreakoutNoFrameskip-v4"
    train_env = gym.make(env_name)
    # train_env = Monitor(env=atari, filename=log_dir)
    train_env = AtariWrapper(env=train_env)
    train_env = FrameStack(train_env, n_stack=4)

    o = train_env.reset()
    print(o.shape)

    for _ in range(400):
        o, r, d, i = train_env.step(train_env.action_space.sample())
        print(o.shape, r, d, i)
        if d:
            o = train_env.reset()
            print(o.shape)
