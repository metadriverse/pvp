import time

# pip install keyboard
import keyboard
from gym import Wrapper
import numpy as np
from collections import defaultdict
import pygame


class HumanInTheLoopAtariWrapper(Wrapper):
    def __init__(self, env, enable_human=False, enable_render=False, mock_human_behavior=False):
        super(HumanInTheLoopAtariWrapper, self).__init__(env=env)

        self.total_cost = 0
        self.takeover = False

        self.enable_human = enable_human
        self.enable_render = enable_render
        self.mock_human_behavior = mock_human_behavior
        if enable_human:
            assert enable_render
        # Key initialization
        if enable_human:
            keyboard.hook(self.discrete_key_detect)

        self.keyboard_action = 0
        self.pause = False
        self.keypress = False

        self.last_time = time.time()
        key_list = env.unwrapped.get_action_meanings()
        self.RIGHT = key_list.index("RIGHT") if "RIGHT" in key_list else 0
        self.LEFT = key_list.index("LEFT") if "LEFT" in key_list else 0
        self.UP = key_list.index("UP") if "UP" in key_list else 0
        self.DOWN = key_list.index("DOWN") if "DOWN" in key_list else 0
        self.NOOP = key_list.index("NOOP") if "NOOP" in key_list else 0

        self.human_in_the_loop_ep_infos = defaultdict(list)
        self.info_keywords = (
            "cost", "total_cost", "takeover_cost", "total_takeover_cost", "raw_action", "takeover", "takeover_start"
        )
        self.displaytext = "Agent behavior"
        self.actionmapping = {0: "No Op", 1: "Right", 2: "Left", 3: "?"}
        # pygame.init()
        # self.display_surface = pygame.display.set_mode((200, 200))
        # pygame.display.set_caption('User Input')
        # self.font = pygame.font.Font('freesansbold.ttf', 32)

    # def rendertext(self, text):
    #     text = self.font.render(text, True, (0, 0, 128), (255,255,255))
    #     textRect = text.get_rect()
    #     textRect.center = (100,100)
    #     self.display_surface.fill((255,255,255))
    #     self.display_surface.blit(text, textRect)
    # pygame.display.flip()
    def step(self, a):
        # discrete: 1 static, 2 right, 3 left
        while self.pause:
            self.pause = True

        # live = self.ale.lives()

        if not self.enable_human and self.mock_human_behavior:
            self.keypress = np.random.uniform(0, 1) < 0.2

        cost = 0.1 if a != self.keyboard_action else 0
        behavior_action = self.keyboard_action if self.keypress else a
        o, r, d, i = super(HumanInTheLoopAtariWrapper, self).step(behavior_action)
        takeover_start = self.keypress and not self.takeover
        if self.keypress:
            # self.rendertext("Takeover: " + str(self.keyboard_action))
            self.displaytext = "Takeover: " + self.actionmapping[self.keyboard_action]
        else:
            self.displaytext = "Agent: " + self.actionmapping[a]
        i["cost"] = cost
        i["total_cost"] = self.total_cost
        i["takeover_cost"] = cost
        i["total_takeover_cost"] = self.total_cost
        i["raw_action"] = [behavior_action]
        i["takeover_start"] = True if takeover_start else False
        i["takeover"] = True if self.keypress else False
        self.takeover = self.keypress

        if self.enable_render:
            self.render("human")

        if self.enable_render:
            now = time.time()
            time.sleep(max(0.0, 0.05 - (now - self.last_time)))
            self.last_time = now

        # "info_keywords" is defined in Monitor
        for key in self.info_keywords:
            self.human_in_the_loop_ep_infos[key].append(i[key])

        if "episode" in i:
            # Fix the Monitor bug!
            assert d

            for key in self.info_keywords:
                i["episode"]["ep_" + key] = np.mean(self.human_in_the_loop_ep_infos[key])

            self.human_in_the_loop_ep_infos.clear()

        return o, r, d, i

    def reset(self):
        self.total_cost = 0
        self.takeover = False
        return super(HumanInTheLoopAtariWrapper, self).reset()

    def discrete_key_detect(self, x):
        if keyboard.is_pressed("a"):
            self.keypress = True
            self.keyboard_action = self.LEFT
        elif keyboard.is_pressed("d"):
            self.keypress = True
            self.keyboard_action = self.RIGHT
        elif keyboard.is_pressed("q"):
            self.keypress = True
            self.keyboard_action = 0
        elif keyboard.is_pressed("s"):
            self.keypress = True
            self.keyboard_action = self.DOWN
        elif keyboard.is_pressed("w"):
            self.keypress = True
            self.keyboard_action = self.UP
        elif keyboard.is_pressed("space"):
            self.keypress = True
            self.keyboard_action = 1
        elif keyboard.is_pressed("f"):
            self.keypress = True
            self.keyboard_action = self.NOOP
        elif keyboard.is_pressed("p"):
            self.keypress = False
            self.pause = not self.pause
        else:
            self.keypress = False

    def puttext(self, image, text):
        import cv2
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 410)
        fontScale = 1
        color = (0, 0, 255)
        thickness = 2
        return cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

    def render(self, mode="human", **kwargs):
        if mode == "human":  # Hack to increase window size
            from gym.envs.classic_control import rendering
            import cv2
            img = super(HumanInTheLoopAtariWrapper, self).render("rgb_array", **kwargs)
            if self.env.viewer is None:
                self.env.viewer = rendering.SimpleImageViewer()
            scale = 2
            resized = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_LINEAR)
            # cv2.imshow("test1",resized)
            resized = self.puttext(resized, self.displaytext)
            # cv2.imshow("test0", resized)
            # cv2.waitKey(500)
            # cv2.imshow("test",resized)
            self.env.viewer.imshow(resized)
            return self.env.viewer.isopen
        else:
            super(HumanInTheLoopAtariWrapper, self).render(mode, **kwargs)


# if __name__ == "__main__":
#     env = DiscreteBreakout()
#     env.reset()
#     while True:
#         o, r, d, i = env.step(keyboard_action)
#         if d:
#             env.reset()
