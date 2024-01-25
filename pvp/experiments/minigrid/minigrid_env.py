"""
Borrowed part of the code from:
https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/manual_control.py
"""
# import gym
import gymnasium as gym
import numpy as np
# from gym import Wrapper

import pygame

from minigrid.envs import EmptyEnv as NativeEmptyEnv

ADDITIONAL_HEIGHT = 200
RULE_WIDTH = 5


def new_render(self):
    img = self.get_frame(self.highlight, self.tile_size, self.agent_pov)

    if self.render_mode == "human":
        img = np.transpose(img, axes=(1, 0, 2))
        if self.render_size is None:
            self.render_size = img.shape[:2]
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.screen_size, self.screen_size + ADDITIONAL_HEIGHT)
            )
            pygame.display.set_caption("minigrid")
        if self.clock is None:
            self.clock = pygame.time.Clock()
        surf = pygame.surfarray.make_surface(img)

        # Create background with mission description
        offset = surf.get_size()[0] * 0.1
        # offset = 32 if self.agent_pov else 64
        bg = pygame.Surface(
            (int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset))
        )
        bg.convert()
        bg.fill((255, 255, 255))
        bg.blit(surf, (offset / 2, 0))

        bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))

        # PZH: Larger font size
        # font_size = 22
        font_size = 50

        text = self.mission
        font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
        text_rect = font.get_rect(text, size=font_size)
        text_rect.center = bg.get_rect().center
        text_rect.y = bg.get_height() - font_size * 1.3
        font.render_to(bg, text_rect, text, size=font_size)
        # self.window.fill((255, 255, 0))

        self.window.blit(bg, (0, 0))

        # ===== PZH: We additionally print something here. =====
        if self.additional_text:
            additional_text_bg = pygame.Surface(
                (self.screen_size, ADDITIONAL_HEIGHT)
            )
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
        # ===== PZH: We additionally print something here. =====

        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()

    elif self.render_mode == "rgb_array":
        return img


class EmptyEnv(NativeEmptyEnv):
    additional_text = ""

    def update_additional_text(self, text):
        self.additional_text = text

    def render(self):
        return new_render(self)


class MiniGridEmpty6x6(EmptyEnv):
    """
    Following:
        register(
            id="MiniGrid-Empty-6x6-v0",
            entry_point="minigrid.envs:EmptyEnv",
            kwargs={"size": 6},
        )
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, size=6, render_mode="human", screen_size=1000, **kwargs)


class MinigridWrapper(gym.Wrapper):
    def __init__(self, env):
        super(MinigridWrapper, self).__init__(env=env)
        self.total_cost = 0
        self.takeover = False
        self.use_render = self.enable_human = env.render_mode == "human"
        self.keyboard_action = None
        self.valid_key_press = False
        # TODO(PZH): We can simply leave only three useful action dimensions here.
        # self.action_space = gym.spaces.Discrete(3)

    def step(self, a):
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
                        else:
                            event.key = pygame.key.name(int(event.key))
                            self.discrete_key_detect(event)
                        should_break = True
                        break

        # TODO: Check all here.
        should_takeover = self.keyboard_action is not None
        cost = 0
        behavior_action = self.keyboard_action if should_takeover else a
        o, r, tm, tc, i = super(MinigridWrapper, self).step(behavior_action)
        d = tm or tc
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
        self.keyboard_action = None
        # ret = super(MinigridWrapper, self).reset()
        ret = self.env.reset()
        if self.use_render:
            pygame.display.set_caption("Reset!")
            self.env.render()
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

    def update_caption(self, agent_action=None):
        if not self.use_render:
            return
        if agent_action is None:
            self.update_additional_text("Waiting agent action...".format(agent_action))
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
            self.update_additional_text("Agent action: {}".format(agent_action))
        if self.use_render:
            self.env.render()

    def close(self):
        self.env.close()
        pygame.quit()
        raise KeyboardInterrupt()


if __name__ == '__main__':
    env = MinigridWrapper(MiniGridEmpty6x6())
    env.reset()
    while True:
        o, r, d, i = env.step(env.action_space.sample())
        if d:
            o = env.reset()
        print(o)
