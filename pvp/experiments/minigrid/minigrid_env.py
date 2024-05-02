"""
Borrowed part of the code from:
https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/manual_control.py

We change the renderer to print text indicating agent's actions.

All environments, except the OldGymWrapper, act like a new gymnasium environment.
"""
import logging
import sys

import gym as old_gym
import gymnasium as gym
import numpy as np
import pygame
from gymnasium.wrappers import FrameStack
from minigrid.envs import EmptyEnv as NativeEmptyEnv
from minigrid.envs import MultiRoomEnv as NativeMultiRoomEnv
from minigrid.wrappers import ImgObsWrapper

from pvp.sb3.common.monitor import Monitor
import dataclasses

# Someone change the logging config. So we have to revert them here.
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger = logging.getLogger(__file__)

ADDITIONAL_HEIGHT = 300
RULE_WIDTH = 5
SCREEN_SIZE = 2000
DEFAULT_TEXT = "Approve: Space/Down | L/R/Forward: Arrow Keys | Toggle: T | Pickup: P | Drop: D | Done: X | Quit: Esc \n"


@dataclasses.dataclass
class ActionMap:
    LEFT: int = 0
    RIGHT: int = 1
    FORWARD: int = 2
    PICKUP: int = 3
    DROP: int = 4
    TOGGLE: int = 5
    DONE: int = 6


@dataclasses.dataclass
class DirMap:
    RIGHT: int = 0
    DOWN: int = 1
    LEFT: int = 2
    UP: int = 3


def new_render(self):
    img = self.get_frame(self.highlight, self.tile_size, self.agent_pov)

    if self.render_mode == "human":
        img = np.transpose(img, axes=(1, 0, 2))
        if self.render_size is None:
            self.render_size = img.shape[:2]
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.screen_size, self.screen_size + ADDITIONAL_HEIGHT))
            pygame.display.set_caption("minigrid")
        if self.clock is None:
            self.clock = pygame.time.Clock()
        surf = pygame.surfarray.make_surface(img)

        # Create background with mission description
        offset = surf.get_size()[0] * 0.1
        # offset = 32 if self.agent_pov else 64
        bg = pygame.Surface((int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset)))
        bg.convert()
        bg.fill((255, 255, 255))
        bg.blit(surf, (offset / 2, 0))

        bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))

        # ===== PZH: We additionally print something here. =====
        font_size = 40
        text = "Mission: {}\n{}".format(self.mission, DEFAULT_TEXT)
        font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
        lines = text.split('\n')
        # Calculate the starting y position
        start_y = bg.get_height() - font_size * 1.5 * (len(lines) - 1)
        for i, line in enumerate(lines):
            # Calculate the y position for each line
            y_position = start_y + (font_size * 1.5 * i)
            # Create a rectangle for the line
            text_rect = font.get_rect(line, size=font_size)
            text_rect.centerx = bg.get_rect().centerx
            text_rect.y = y_position
            # Render the line to the background surface
            font.render_to(bg, text_rect, line, size=font_size)

        # text_rect = font.get_rect(text, size=font_size)
        # text_rect.center = bg.get_rect().center
        # text_rect.y = bg.get_height() - font_size * 1.3
        # font.render_to(bg, text_rect, text, size=font_size)
        # self.window.fill((255, 255, 0))

        self.window.blit(bg, (0, 0))
        font_size = 80
        if self.additional_text:
            additional_text_bg = pygame.Surface((self.screen_size, ADDITIONAL_HEIGHT))
            additional_text_bg.fill((255, 255, 255))
            # Split the text into lines
            lines = self.additional_text.split('\n')
            # Calculate the starting y position
            start_y = font_size * 0.5
            for i, line in enumerate(lines):
                # Calculate the y position for each line
                y_position = start_y + (font_size * 1.5 * i)
                # Create a rectangle for the line
                text_rect = font.get_rect(line, size=font_size)
                text_rect.centerx = additional_text_bg.get_rect().centerx
                text_rect.y = y_position
                # Render the line to the background surface
                font.render_to(additional_text_bg, text_rect, line, size=font_size)
            self.window.blit(additional_text_bg, (0, self.screen_size + RULE_WIDTH))
        # ===== PZH: We additionally print something here DONE. =====

        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()

    elif self.render_mode == "rgb_array":
        return img


class OldGymWrapper(old_gym.Wrapper):
    def step(self, *args, **kwargs):
        o, r, tm, tc, i = super().step(*args, **kwargs)
        return o, r, tm or tc, i

    def reset(self, *args, **kwargs):
        o, i = super().reset(*args, **kwargs)
        return o


class ConcatenateChannel(gym.ObservationWrapper):
    """Convert the observation from shape (4, 7, 7, 3) to (12, 7, 7), in channel-first manner."""
    def __init__(self, env):
        super(ConcatenateChannel, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, dtype=np.uint8, shape=(old_shape[0] * old_shape[-1], *old_shape[1:-1])
        )

    def observation(self, obs):
        assert len(obs.shape) == 4
        obs = np.swapaxes(obs, 1, -1)
        obs = np.reshape(obs, (-1, *obs.shape[2:]))
        return obs


class EmptyEnv(NativeEmptyEnv):
    additional_text = ""

    def update_additional_text(self, text):
        self.additional_text = text

    def render(self):
        self.metadata["render_fps"] = 5
        return new_render(self)


class MultiRoomEnv(NativeMultiRoomEnv):
    additional_text = ""

    def update_additional_text(self, text):
        self.additional_text = text

    def render(self):
        return new_render(self)


class MiniGridEmpty6x6(EmptyEnv):
    """Empty Room.
    Following:
        register(
            id="MiniGrid-Empty-6x6-v0",
            entry_point="minigrid.envs:EmptyEnv",
            kwargs={"size": 6},
        )
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size=6, **kwargs)


class MiniGridEmpty16x16(EmptyEnv):
    """Empty Room.
    Following:
        register(
            id="MiniGrid-Empty-6x6-v0",
            entry_point="minigrid.envs:EmptyEnv",
            kwargs={"size": 6},
        )
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size=16, **kwargs)


class MiniGridMultiRoomN2S4(MultiRoomEnv):
    """Two Room.
    Following:
        register(
            id="MiniGrid-MultiRoom-N2-S4-v0",
            entry_point="minigrid.envs:MultiRoomEnv",
            kwargs={"minNumRooms": 2, "maxNumRooms": 2, "maxRoomSize": 4},
        )
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, minNumRooms=2, maxNumRooms=2, maxRoomSize=4, **kwargs)


class MiniGridMultiRoomN4S5(MultiRoomEnv):
    """Four Room.
    Following:
        register(
            id="MiniGrid-MultiRoom-N4-S5-v0",
            entry_point="minigrid.envs:MultiRoomEnv",
            kwargs={"minNumRooms": 6, "maxNumRooms": 6, "maxRoomSize": 5},
        )
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, minNumRooms=6, maxNumRooms=6, maxRoomSize=5, **kwargs)


class MinigridWrapper(gym.Wrapper):
    def __init__(self, env):
        super(MinigridWrapper, self).__init__(env=env)
        self.total_takeover = 0
        self.total_steps = 0
        self.takeover = False
        self.use_render = self.enable_human = env.render_mode == "human"
        self.keyboard_action = None
        self.valid_key_press = False

    def step(self, a):
        self.total_steps += 1
        self.update_caption(a)

        if self.enable_human:
            should_break = False
            while not should_break:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        # TODO: Deal with this.
                        self.close()
                        should_break = True
                        break
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.close()
                            raise KeyboardInterrupt()
                        else:
                            event.key = pygame.key.name(int(event.key))
                            self.discrete_key_detect(event)
                            if self.valid_key_press:
                                should_break = True
                            else:
                                should_break = False
                        break

        should_takeover = self.keyboard_action is not None
        cost = 0
        behavior_action = self.keyboard_action if should_takeover else a
        o, r, tm, tc, i = super(MinigridWrapper, self).step(behavior_action)
        takeover_start = should_takeover and not self.takeover
        i["cost"] = cost
        i["total_takeover"] = self.total_takeover
        i["takeover_cost"] = cost
        i["raw_action"] = int(behavior_action)
        i["takeover_start"] = True if takeover_start else False
        i["takeover"] = True if should_takeover else False
        i["is_success"] = i["success"] = True if r > 0.0 else False
        self.takeover = should_takeover
        self.valid_key_press = False  # refresh
        self.update_caption(None)  # Set caption to "waiting"
        self.total_takeover += 1 if self.takeover else 0
        return o, r, tm, tc, i

    def reset(self, *args, **kwargs):
        self.takeover = False
        self.keyboard_action = None
        ret = self.env.reset(*args, **kwargs)
        if self.use_render:
            pygame.display.set_caption("Reset!")
            self.env.render()
        return ret

    def discrete_key_detect(self, event):
        if event.key == "left":
            self.valid_key_press = True
            self.keyboard_action = self.env.actions.left
        elif event.key == "right":
            self.valid_key_press = True
            self.keyboard_action = self.env.actions.right
        elif event.key == "up":
            self.valid_key_press = True
            self.keyboard_action = self.env.actions.forward
        elif event.key == "space" or event.key == "down":  # Space/Down means allowing agent action!
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
            logger.warning("Find unknown key press event: {}! Please press again!".format(event.key))

    def update_caption(self, agent_action=None):
        suffix = "Human: {}, Total: {} ({:.1f}%)".format(
            self.total_takeover, self.total_steps, self.total_takeover / self.total_steps * 100
        )
        if not self.use_render:
            return
        if agent_action is None:
            self.update_additional_text("Waiting agent action...\n{}".format(suffix))
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
            self.update_additional_text("Agent action: {}\n{}".format(agent_action, suffix))
        if self.use_render:
            self.env.render()

    def close(self):
        self.env.close()
        pygame.quit()

    def seed(self, seed):
        """Forward compatibility to gymnasium"""
        self.env.reset(seed=seed)


class MinigridWrapperWithFakeHuman(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env=env)
        self.total_takeover = 0
        self.total_steps = 0
        self.takeover = False
        # self.use_render = self.enable_human = env.render_mode == "human"
        # self.keyboard_action = None
        # self.valid_key_press = False

    def get_expert_action(self):

        # Make fake action:
        x, y = self.env.agent_pos
        dir = self.env.agent_dir
        RIGHT_MOST = self.env.grid.width - 2

        if x != RIGHT_MOST:
            # Try to move to the right
            if dir in [DirMap.LEFT, DirMap.UP]:
                # Turn right
                expert_action = ActionMap.RIGHT
            elif dir == DirMap.RIGHT:
                # Move forward
                expert_action = ActionMap.FORWARD
            else:
                # Turn left
                expert_action = ActionMap.LEFT
        else:
            # Try to move down
            if dir in [DirMap.UP, DirMap.RIGHT]:
                # Turn right
                expert_action = ActionMap.RIGHT
            elif dir == DirMap.DOWN:
                # Move forward
                expert_action = ActionMap.FORWARD
            else:
                # Turn left
                expert_action = ActionMap.LEFT
        return expert_action

    def step(self, a):
        self.total_steps += 1

        expert_action = self.get_expert_action()
        if a == expert_action:
            behavior_action = a
            should_takeover = False
        else:
            behavior_action = expert_action
            should_takeover = True

        cost = 0

        o, r, tm, tc, i = super().step(behavior_action)
        takeover_start = should_takeover and not self.takeover
        i["cost"] = cost
        i["total_takeover"] = self.total_takeover
        i["takeover_cost"] = cost
        i["raw_action"] = int(behavior_action)
        i["takeover_start"] = True if takeover_start else False
        i["takeover"] = True if should_takeover else False
        i["is_success"] = i["success"] = True if r > 0.0 else False
        self.takeover = should_takeover
        # self.valid_key_press = False  # refresh
        # self.update_caption(None)  # Set caption to "waiting"
        self.total_takeover += 1 if self.takeover else 0
        return o, r, tm, tc, i

    def reset(self, *args, **kwargs):
        self.takeover = False
        # self.keyboard_action = None
        ret = self.env.reset(*args, **kwargs)
        # if self.use_render:
        #     pygame.display.set_caption("Reset!")
        #     self.env.render()
        return ret

    def close(self):
        self.env.close()
        pygame.quit()

    def seed(self, seed):
        """Forward compatibility to gymnasium"""
        self.env.reset(seed=seed)


class MinigridWrapperWithFakeHumanAndHumanFailureDemo(gym.Wrapper):
    def __init__(self, env, failed_step=50, failed_prob=0.1):
        super().__init__(env=env)

        self.total_takeover = 0
        self.total_steps = 0
        self.takeover = False

        self.episode_step = 0
        self.failed_step = failed_step
        self.failed_prob = failed_prob
        self.start_failed = False
        self.in_failed_step = 0

    def close(self):
        self.env.close()
        pygame.quit()

    def seed(self, seed):
        """Forward compatibility to gymnasium"""
        self.env.reset(seed=seed)

    def get_expert_action(self):

        # Make fake action:
        x, y = self.env.agent_pos
        dir = self.env.agent_dir
        RIGHT_MOST = self.env.grid.width - 2

        if x != RIGHT_MOST:
            # Try to move to the right
            if dir in [DirMap.LEFT, DirMap.UP]:
                # Turn right
                expert_action = ActionMap.RIGHT
            elif dir == DirMap.RIGHT:
                # Move forward
                expert_action = ActionMap.FORWARD
            else:
                # Turn left
                expert_action = ActionMap.LEFT
        else:
            # Try to move down
            if dir in [DirMap.UP, DirMap.RIGHT]:
                # Turn right
                expert_action = ActionMap.RIGHT
            elif dir == DirMap.DOWN:
                # Move forward
                expert_action = ActionMap.FORWARD
            else:
                # Turn left
                expert_action = ActionMap.LEFT
        return expert_action

    def step(self, a):
        if not self.start_failed and self.in_failed_step != self.failed_step:
            if np.random.uniform() < self.failed_prob:
                self.start_failed = True
                self.in_failed_step = 0
                # print("Start failure demo")
        expert_action = self.get_expert_action()
        if self.start_failed:
            self.in_failed_step += 1
            # print("in_failed_step: ", self.in_failed_step)
            if self.in_failed_step == self.failed_step:
                # print("end failure demo")
                self.start_failed = False
            wrong_action = {ActionMap.LEFT, ActionMap.RIGHT, ActionMap.FORWARD} - {expert_action}
            wrong_action = np.random.choice(list(wrong_action))
            behavior_action = int(wrong_action)
            should_takeover = True
            # print("using wrong action: ", a)
        else:
            behavior_action = expert_action
            should_takeover = False
            # print("using expert_action action: ", a)

        self.total_steps += 1

        cost = 0
        # print("behavior action: ", behavior_action)
        o, r, tm, tc, i = super().step(behavior_action)
        takeover_start = should_takeover and not self.takeover
        i["cost"] = cost
        i["total_takeover"] = self.total_takeover
        i["takeover_cost"] = cost
        i["raw_action"] = int(behavior_action)
        i["takeover_start"] = True if takeover_start else False
        i["takeover"] = True if should_takeover else False
        i["is_success"] = i["success"] = True if r > 0.0 else False
        self.takeover = should_takeover
        # self.valid_key_press = False  # refresh
        # self.update_caption(None)  # Set caption to "waiting"
        self.total_takeover += 1 if self.takeover else 0
        return o, r, tm, tc, i

    def reset(self, *args, **kwargs):
        print("reset")
        # out = super().reset(*args, **kwargs)

        # def reset(self, *args, **kwargs):
        self.takeover = False
        out = self.env.reset(*args, **kwargs)

        self.episode_step = 0
        self.start_failed = False
        self.in_failed_step = 0
        return out


def wrap_minigrid_env(env_class, enable_takeover, use_fake_human=False, use_fake_human_with_failure=False):
    if enable_takeover:
        env = env_class(render_mode="human", screen_size=SCREEN_SIZE)
    else:
        env = env_class()

    if use_fake_human:
        if use_fake_human_with_failure:
            env = MinigridWrapperWithFakeHumanAndHumanFailureDemo(env)
        else:
            env = MinigridWrapperWithFakeHuman(env)
    else:
        env = MinigridWrapper(env)
    env = ImgObsWrapper(env)
    env = FrameStack(env, num_stack=4)
    env = ConcatenateChannel(env)
    env = OldGymWrapper(env)
    return env


if __name__ == '__main__':
    # env = MinigridWrapper(MiniGridEmpty6x6())
    env = wrap_minigrid_env(
        MiniGridEmpty16x16, enable_takeover=True, use_fake_human=True, use_fake_human_with_failure=True
    )
    env = Monitor(env)
    env.reset()
    print(env.observation_space)
    while True:
        # o, r, d, i = env.step(env.action_space.sample())
        o, r, d, i = env.step(ActionMap.RIGHT)
        if d:
            o = env.reset()
        # print(o.shape)
