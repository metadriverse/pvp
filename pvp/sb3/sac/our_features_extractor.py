import gym.spaces
import torch as th
from torch import nn

from pvp.sb3.common.torch_layers import BaseFeaturesExtractor
from pvp.utils.utils import same_padding

predefined_filters = {
    (240, 320): [
        [16, [12, 16], [7, 9]],
        [32, [6, 6], 4],
        [256, [9, 9], 1],
    ],
    (180, 320): [
        [16, [9, 16], [5, 9]],  # output: 36, 36
        [32, [3, 3], 2],  # output: 18, 18
        [64, [3, 3], 2],  # output: 9, 9
        [128, [3, 3], 3],  # output: 3, 3
        [256, [3, 3], 3],  # output: 1, 1
    ],
    (84, 84): [
        [16, [4, 4], 3],  # output: 28, 28
        [32, [3, 3], 2],  # output: 14, 14
        [64, [3, 3], 2],  # output: 7, 7
        [128, [3, 3], 2],  # output: 4, 4
        [256, [4, 4], 4],  # output: 1, 1
    ],

    # (42, 42): [
    #     [32, [4, 4], 3],  # output: 14, 14
    #     [64, [3, 3], 2],  # output: 7, 7
    #     [128, [3, 3], 2],  # output: 4, 4
    #     [256, [4, 4], 4],  # output: 1, 1
    # ],

    # PZH: A very tiny network!
    (42, 42): [
        [16, [4, 4], 3],  # output: 14, 14
        [32, [3, 3], 2],  # output: 7, 7
        [64, [3, 3], 2],  # output: 4, 4
        [128, [4, 4], 4],  # output: 1, 1
    ],
}


class OurFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(OurFeaturesExtractor, self).__init__(observation_space, features_dim)
        if isinstance(observation_space, gym.spaces.Box):
            obs_shape = observation_space.shape
            self.use_dict_obs_space = False
        else:
            obs_shape = observation_space["image"].shape
            self.use_dict_obs_space = True
        input_image_size = obs_shape[1:]
        self.filters = predefined_filters[input_image_size]
        layers = []
        input_size = obs_shape[0]
        for output_size, kernel, stride in self.filters:
            padding, input_image_size = same_padding(input_image_size, kernel, stride)
            layers.append(nn.ZeroPad2d(padding))
            layers.append(nn.Conv2d(input_size, output_size, kernel_size=kernel, stride=stride))
            layers.append(nn.ReLU())
            input_size = output_size
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        if self.use_dict_obs_space:
            obs_feat = self.cnn(observations["image"])
            other_feat = observations["speed"]
            ret = th.cat([obs_feat, other_feat], dim=1)
        else:
            ret = self.cnn(observations)
        assert ret.shape[-1] == self._features_dim
        return ret
