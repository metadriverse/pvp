import copy
import os
from typing import Any, Dict, Optional

import cv2
import numpy as np
from easydict import EasyDict

from pvp.experiments.carla.di_drive.core.utils.others.config_helper import deep_merge_dicts
from pvp.experiments.carla.di_drive.core.utils.others.image_helper import GifMaker, VideoMaker, show_image

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def draw_texts(data_dict, canvas, color=BLACK, thick=2):
    def _write(text, i, j, fontsize=0.4, choose_color=color):
        rows = [x * (canvas.shape[0] // 30) for x in range(10 + 1)]
        cols = [x * (canvas.shape[1] // 15) for x in range(10 + 1)]
        cv2.putText(
            canvas, text, (cols[j], rows[i]), cv2.FONT_HERSHEY_SIMPLEX, fontsize, choose_color, thick, cv2.LINE_AA
        )

    if canvas.shape[0] > 600:
        fontsize = canvas.shape[0] * 0.0008
    else:
        fontsize = 0.4

    left_text_pos = 1
    left_text_horizontal_pos = 3

    _write('Exit: Press X Button', left_text_pos, left_text_horizontal_pos, fontsize=fontsize)
    left_text_pos += 1

    if 'command' in data_dict:
        _command = {
            -1: 'VOID',
            1: 'LEFT',
            2: 'RIGHT',
            3: 'STRAIGHT',
            4: 'FOLLOW',
            5: 'CHANGE LEFT',
            6: 'CHANGE RIGHT',
        }.get(data_dict['command'], '???')
        _write('Command: ' + _command, left_text_pos, left_text_horizontal_pos, fontsize=fontsize)
        left_text_pos += 1
    if 'agent_state' in data_dict:
        _state = {
            -1: 'VOID',
            1: 'NAVIGATING',
            2: 'BLOCKED_BY_VEHICLE',
            3: 'BLOCKED_BY_WALKER',
            4: 'BLOCKED_RED_LIGHT',
            5: 'BLOCKED_BY_BIKE',
        }.get(data_dict['agent_state'], '???')
        _write('Agent State: ' + _state, left_text_pos, left_text_horizontal_pos, fontsize=fontsize)
        left_text_pos += 1
    if 'speed' in data_dict:
        text = 'Speed: {:04.1f}'.format(data_dict['speed_kmh'])
        # if 'speed_limit' in data_dict:
        #     text += '/{:.1f}'.format(data_dict['speed_limit'] * 3.6)
        text += " km/h"
        _write(text, left_text_pos, left_text_horizontal_pos, fontsize=fontsize)
        left_text_pos += 1
    if 'steer' in data_dict and 'throttle' in data_dict and 'brake' in data_dict:
        _write('Steer: {:.3f}'.format(data_dict['steer']), left_text_pos, left_text_horizontal_pos, fontsize=fontsize)
        _write(
            'Throttle: {:.3f}'.format(data_dict['throttle']),
            left_text_pos + 1,
            left_text_horizontal_pos,
            fontsize=fontsize
        )
        _write(
            'Brake: {:.3f}'.format(data_dict['brake']), left_text_pos + 2, left_text_horizontal_pos, fontsize=fontsize
        )
        left_text_pos += 3
    if data_dict.get('takeover', False):
        if color == BLACK:
            _write('Taking Over!', left_text_pos, left_text_horizontal_pos, fontsize=fontsize)
        else:
            _write('Taking Over!', left_text_pos, left_text_horizontal_pos, fontsize=fontsize, choose_color=(0, 255, 0))
        left_text_pos += 1

    right_text_pos = 1

    _write('Pause: Press TRIANGLE Button', right_text_pos, 9, fontsize=fontsize)
    right_text_pos += 1

    if 'total_step' in data_dict:
        _write(
            'Total Step: {} ({:02d}:{:04.1f})'.format(
                data_dict["total_step"], int(data_dict["total_time"] // 60), data_dict["total_time"] % 60
            ),
            right_text_pos,
            9,
            fontsize=fontsize
        )
        right_text_pos += 1
    # if 'total_lights' in data_dict and 'total_lights_ran' in data_dict:
    #     text = 'Lights Ran: %d/%d' % (data_dict['total_lights_ran'], data_dict['total_lights'])
    #     _write(text, right_text_pos, 9, fontsize=fontsize)
    #     right_text_pos += 1
    if 'takeover_rate' in data_dict:
        _write(
            'Takeover Rate: {:04.1f} %'.format(100 * data_dict["takeover_rate"]), right_text_pos, 9, fontsize=fontsize
        )
        right_text_pos += 1
    if 'distance_to_go' in data_dict:
        text = 'Distance to go: %.1f' % data_dict['distance_to_go']
        if 'distance_total' in data_dict:
            text += '/{:.1f} ({:04.1f} %)'.format(
                data_dict['distance_total'], 100 * (1 - data_dict['distance_to_go'] / data_dict['distance_total'])
            )
        _write(text, right_text_pos, 9, fontsize=fontsize)
        right_text_pos += 1
    if 'tick' in data_dict:
        text = 'Step: %d' % data_dict['tick']
        if 'end_timeout' in data_dict:
            text += '/%d' % data_dict['end_timeout']
        _write(text, right_text_pos, 9, fontsize=fontsize)
        right_text_pos += 1
    if 'reward' in data_dict:
        text = 'Reward: %.02f' % data_dict['reward']
        if 'episode_reward' in data_dict:
            text += '/%.02f' % data_dict['episode_reward']
        _write(text, right_text_pos, 9, fontsize=fontsize)
        right_text_pos += 1
    if 'FPS' in data_dict:
        _write('FPS: %04.1f' % data_dict['FPS'], right_text_pos, 9, fontsize=fontsize)
        right_text_pos += 1
    if data_dict.get('stuck', False):
        _write('Stuck!', right_text_pos, 9, fontsize=fontsize)
        right_text_pos += 1
    if data_dict.get('ran_light', False):
        _write('Ran light!', right_text_pos, 9, fontsize=fontsize)
        right_text_pos += 1
    if data_dict.get('off_road', False):
        _write('Off road!', right_text_pos, 9, fontsize=fontsize)
        right_text_pos += 1
    if data_dict.get('wrong_direction', False):
        if color == BLACK:
            _write('Wrong direction!', right_text_pos, 9, fontsize=fontsize)
        else:
            _write('Wrong direction!', right_text_pos, 9, fontsize=fontsize, choose_color=(255, 0, 0))
        right_text_pos += 1


class Visualizer(object):
    """
    Visualizer is used to visualize sensor data and print info during running.
    It can be used to show a sensor image on screen, save a gif or video file.

    :Arguments:
        - cfg (Dict): Config dict.

    :Interfaces: init, paint, run_visualize, done
    """
    _name = None
    _canvas = None
    _gif_maker = None
    _video_maker = None
    _already_show_window = False
    config = dict(
        show_text=True,
        outputs=list(),
        save_dir='',
        frame_skip=0,
        min_size=400,
        location=None,
    )

    def __init__(self, cfg: Dict) -> None:
        if 'cfg_type' not in cfg:
            self._cfg = self.__class__.default_config()
            self._cfg = deep_merge_dicts(self._cfg, cfg)
        else:
            self._cfg = cfg

        self._text = self._cfg.show_text
        self._outputs = self._cfg.outputs
        self._save_dir = self._cfg.save_dir

        self._count = 0
        self._frame_skip = self._cfg.frame_skip

        if self._save_dir != '':
            os.makedirs(self._save_dir, exist_ok=True)

    def init(self, name: str) -> None:
        """
        Initlaize visualizer with provided name.

        :Arguments:
            - name (str): Name for window or file.
        """
        self._name = "CARLA Simulator 0.9.10.1"
        # cv2.waitKey(1)
        if 'gif' in self._outputs:
            self._gif_maker = GifMaker()
        if 'video' in self._outputs:
            self._video_maker = VideoMaker()
            self._video_maker.init(self._save_dir, self._name)

    def paint(self, image: Any, data_dict: Optional[Dict] = None, monitor_index=0) -> None:
        """
        Paint canvas with observation images and data.

        :Arguments:
            - image: Rendered image.
            - data_dict(Dict, optional): data dict containing information, state, action and so on
        """
        if data_dict is None:
            data_dict = {}
        self._canvas = np.uint8(image.copy())

        h, w = self._canvas.shape[:2]
        if min(h, w) < self._cfg.min_size:
            rate = self._cfg.min_size / min(h, w)
            self._canvas = resize_birdview(self._canvas, rate)

        if not self._already_show_window:
            move_window(
                self._name,
                self._cfg["location"],
                image_x=self._canvas.shape[1],
                image_y=self._canvas.shape[0],
                monitor_index=monitor_index
            )
            self._already_show_window = True

        if not self._text:
            return

        draw_texts(data_dict, self._canvas, BLACK, thick=5)
        draw_texts(data_dict, self._canvas, WHITE, thick=2)

    def run_visualize(self) -> None:
        """
        Run one step visualizer. Update file handler or show screen.
        """
        if self._canvas is None:
            return
        self._count += 1
        if self._count > self._frame_skip:
            if 'gif' in self._outputs:
                self._gif_maker.add(self._name, self._canvas)
            if 'video' in self._outputs:
                self._video_maker.add(self._canvas)
            self._count = 0
        if 'show' in self._outputs:
            show_image(self._canvas, name=self._name)

    def done(self) -> None:
        """
        Save file or release file writter, destroy windows.
        """
        if self._gif_maker is not None:
            self._gif_maker.save(self._name, self._save_dir, self._name + '.gif')
            self._gif_maker.clear(self._name)
        if self._video_maker is not None:
            self._video_maker.clear()
        # if 'show' in self._outputs:
        #     cv2.destroyAllWindows()

    @property
    def canvas(self):
        return self._canvas

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(cls.config)
        cfg.cfg_type = cls.__name__ + 'Config'
        return copy.deepcopy(cfg)


def resize_birdview(img, rate):
    assert len(img.shape) == 3
    img_res_list = []
    for i in range(img.shape[2]):
        img_slice = img[..., i]
        img_slice_res = cv2.resize(img_slice, None, fx=rate, fy=rate, interpolation=cv2.INTER_NEAREST)
        img_res_list.append(img_slice_res)
    img_res = np.stack(img_res_list, axis=2)
    return img_res


def move_window(name, location, image_x, image_y, monitor_index=0):
    """monitor_index starts by 0."""
    from screeninfo import get_monitors
    monitors = get_monitors()
    assert monitor_index < len(monitors)
    current_monitor = list(get_monitors())[monitor_index]
    cv2.namedWindow(name)
    if location == "upper left":
        cv2.moveWindow(name, current_monitor.x, current_monitor.y)
    elif location == "lower right":
        cv2.moveWindow(
            name, current_monitor.x + current_monitor.width - image_x,
            current_monitor.y + current_monitor.height - image_y
        )
    elif (location is None) or (location == "center"):
        cv2.moveWindow(
            name, int(current_monitor.x + (current_monitor.width - image_x) / 2),
            int(current_monitor.y + (current_monitor.height - image_y) / 2)
        )
    else:
        raise ValueError("Unknown location: {}".format(location))
