import platform

import evdev
import pygame
from evdev import ecodes, InputDevice

is_windows = "Win" in platform.system()


class SteeringWheelController:
    RIGHT_SHIFT_PADDLE = 4
    LEFT_SHIFT_PADDLE = 5
    STEERING_MAKEUP = 1.5

    def __init__(self, disable=False):
        self.disable = disable
        if not self.disable:
            pygame.display.init()
            pygame.joystick.init()
            assert pygame.joystick.get_count() > 0, "Please connect joystick or use keyboard input"
            print("Successfully Connect your Joystick!")

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

        hat = self.joystick.get_hat(0)
        self.button_up = True if hat[-1] == 1 else False
        self.button_down = True if hat[-1] == -1 else False
        self.button_left = True if hat[0] == -1 else False
        self.button_right = True if hat[0] == 1 else False

        self.feedback(speed_kmh)

        return [-steering * self.STEERING_MAKEUP, (throttle - brake)]

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


if __name__ == '__main__':
    device = evdev.list_devices()[0]
    evtdev = InputDevice(device)
    val = 65535  # val \in [0,65535]
    while True:
        # try:
        evtdev.write(ecodes.EV_FF, ecodes.FF_AUTOCENTER, val)
        # except:
        #     evtdev.write(ecodes.EV_FF, ecodes.FF_AUTOCENTER, 0)
        #     break
