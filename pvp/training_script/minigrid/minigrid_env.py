import cv2
import gym
import gym_minigrid
import matplotlib.pyplot as plt
from gym import Wrapper
from gym_minigrid.window import Window

# from gym_minigrid.wrappers import *

gym_minigrid


class GrayScaleWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(env.observation_space.shape[0], env.observation_space.shape[1], 1),
            dtype=env.observation_space.dtype
        )

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return frame[:, :, None]


class MinigridWrapper(Wrapper):
    def __init__(self, env, enable_human=False, enable_render=False):
        super(MinigridWrapper, self).__init__(env=env)
        self.total_cost = 0
        self.takeover = False
        self.keyboard_action = None
        self.valid_key_press = False

        self.enable_human = enable_human
        if enable_human:
            assert enable_render
        if enable_render:
            self.window = Window('gym_minigrid - ' + self.env.spec.id)
            if enable_human:
                self.window.reg_key_handler(self.discrete_key_detect)
        else:
            self.window = None

        # xxx: We can simply leave three useful action dimensions here.
        #  but for consistency for future complex grid env, we just leave the nextli
        # self.action_space = gym.spaces.Discrete(3)

    def step(self, a):
        self.update_caption(a)

        if self.enable_human:
            button = plt.waitforbuttonpress(0.03)
            while (not button) or (not self.valid_key_press):
                button = plt.waitforbuttonpress(0.03)

        should_takeover = self.keyboard_action is not None
        # if should_takeover and a != self.keyboard_action:
        #     cost = 0.1
        # else:
        #     cost = 0
        cost = 0
        behavior_action = self.keyboard_action if should_takeover else a
        o, r, d, i = super(MinigridWrapper, self).step(behavior_action)
        takeover_start = should_takeover and not self.takeover
        i["cost"] = cost
        i["total_cost"] = self.total_cost
        i["takeover_cost"] = cost
        i["total_takeover_cost"] = self.total_cost
        i["raw_action"] = [int(behavior_action)]
        i["takeover_start"] = True if takeover_start else False
        i["takeover"] = True if should_takeover and self.takeover else False
        i["is_success"] = i["success"] = True if r > 0.0 else False
        self.takeover = should_takeover
        self.valid_key_press = False  # refresh

        self.update_caption(None)  # Set caption to "waiting"

        return o, r, d, i

    def reset(self):
        self.total_cost = 0
        self.takeover = False
        ret = super(MinigridWrapper, self).reset()

        if self.window is not None:
            self.window.set_caption("Reset!")
            self.redraw()

        return ret

    def discrete_key_detect(self, event):
        print("Find key press event!", event.key)
        if event.key == "left":
            self.valid_key_press = True
            self.keyboard_action = self.env.actions.left
        elif event.key == "right":
            self.valid_key_press = True
            self.keyboard_action = self.env.actions.right
        elif event.key == "up":
            self.valid_key_press = True
            self.keyboard_action = self.env.actions.forward
        elif event.key == " " or event.key == "down":  # Space/Down means allowing agent action!
            self.valid_key_press = True
            self.keyboard_action = None

        elif event.key == "p":
            # Pick up an object
            self.valid_key_press = True
            self.keyboard_action = self.env.actions.pickup

        elif event.key == "d":
            # Drop an object
            self.valid_key_press = True
            self.keyboard_action = self.env.actions.drop

        elif event.key == "t":
            # Toggle/activate an object
            self.valid_key_press = True
            self.keyboard_action = self.env.actions.toggle

        elif event.key == "x":
            # Done completing task
            self.valid_key_press = True
            self.keyboard_action = self.env.actions.done

        else:
            self.valid_key_press = False

    def redraw(self):
        if self.window is None:
            return

        img = self.env.render('rgb_array', tile_size=32)
        self.window.show_img(img)

    def update_caption(self, agent_action=None):
        if self.window is None:
            return

        if agent_action is None:
            self.window.set_caption("Waiting ...".format(agent_action))
        else:
            if agent_action < 7:
                agent_action = {
                    0: "Turning Left",
                    1: "Turning Right",
                    2: "Forward",
                    3: "Pickup",
                    4: "Drop",
                    5: "Toggle/Activate Object",
                    6: "Done Complete Task",
                }[agent_action]
            else:
                agent_action = "Invalid action: {}".format(agent_action)
            self.window.set_caption("Agent action: {}".format(agent_action))

        img = self.env.render('rgb_array', tile_size=32)
        self.window.show_img(img)

        # We have to redraw image after changing the caption
        self.redraw()





import gym
# import torch
from collections import deque, defaultdict
from gym import spaces
import numpy as np
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX

# Copied from: https://github.com/facebookresearch/impact-driven-exploration/blob/877c4ea530cc0ca3902211dba4e922bf8c3ce276/src/env_utils.py#L38

class FullyObsCustomWrapper(gym.Wrapper):
    def __init__(self, env, fix_seed=False, env_seed=1):
        super(FullyObsCustomWrapper, self).__init__(env)
        self.episode_return = None
        self.episode_step = None
        self.episode_win = None
        self.fix_seed = fix_seed
        self.env_seed = env_seed

    def get_partial_obs(self):
        return self.gym_env.env.env.gen_obs()['image']

    def initial(self):
        initial_reward = torch.zeros(1, 1)
        # self.episode_return = torch.zeros(1, 1)
        # self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        # self.episode_win = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.ones(1, 1, dtype=torch.uint8)
        if self.fix_seed:
            self.gym_env.seed(seed=self.env_seed)
        initial_frame = _format_observation(self.gym_env.reset())
        partial_obs = _format_observation(self.get_partial_obs())

        if self.gym_env.env.env.carrying:
            carried_col, carried_obj = torch.LongTensor([[COLOR_TO_IDX[self.gym_env.env.env.carrying.color]]]), torch.LongTensor([[OBJECT_TO_IDX[self.gym_env.env.env.carrying.type]]])
        else:
            carried_col, carried_obj = torch.LongTensor([[5]]), torch.LongTensor([[1]])

        return dict(
            frame=initial_frame,
            reward=initial_reward,
            done=initial_done,
            # episode_return=self.episode_return,
            # episode_step=self.episode_step,
            # episode_win=self.episode_win,
            carried_col = carried_col,
            carried_obj = carried_obj,
            partial_obs=partial_obs
        )

    def step(self, action):
        frame, reward, done, _ = self.gym_env.step(action.item())

        self.episode_step += 1
        episode_step = self.episode_step

        self.episode_return += reward
        episode_return = self.episode_return

        if done and reward > 0:
            self.episode_win[0][0] = 1
        else:
            self.episode_win[0][0] = 0
        episode_win = self.episode_win

        if done:
            if self.fix_seed:
                self.gym_env.seed(seed=self.env_seed)
            frame = self.gym_env.reset()
            # self.episode_return = torch.zeros(1, 1)
            # self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
            # self.episode_win = torch.zeros(1, 1, dtype=torch.int32)

        frame = _format_observation(frame)
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)
        partial_obs = _format_observation(self.get_partial_obs())

        if self.gym_env.env.env.carrying:
            carried_col, carried_obj = torch.LongTensor([[COLOR_TO_IDX[self.gym_env.env.env.carrying.color]]]), torch.LongTensor([[OBJECT_TO_IDX[self.gym_env.env.env.carrying.type]]])
        else:
            carried_col, carried_obj = torch.LongTensor([[5]]), torch.LongTensor([[1]])


        return dict(
            frame=frame,
            reward=reward,
            done=done,
            episode_return=episode_return,
            episode_step = episode_step,
            episode_win = episode_win,
            carried_col = carried_col,
            carried_obj = carried_obj,
            partial_obs=partial_obs
        )

    def get_full_obs(self):
        env = self.gym_env.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])
        return full_grid

    def close(self):
        self.gym_env.close()



if __name__ == '__main__':

    # Preliminary:
    #  pip install gym-minigrid

    # Github repo: Farama-Foundation/gym-minigrid
    # Key idea: stop at each step!
    # Keys:
    #   up arrow - move forward
    #   left/right arrows - rotate left/right
    #   space - allow agent action

    env_name = 'MiniGrid-Empty-6x6-v0'

    # Don't forget this wrapper!
    env = MinigridWrapper(gym.make(env_name))

    env.reset()
    while True:
        o, r, d, i = env.step(env.action_space.sample())
        if d:
            o = env.reset()
        print(o)
