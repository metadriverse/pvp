import gym
import torch as th

from pvp.sb3.common.preprocessing import is_image_space
from pvp.sb3.common.torch_layers import BaseFeaturesExtractor

# Model from: https://github.com/lcswillems/rl-starter-files/blob/e604b36915a13e25ac8a8a912f9a9a15e2d4a170/model.py


def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / th.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class MinigridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(MinigridCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False)
        # n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 16, (2, 2)), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)), nn.ReLU(), nn.Conv2d(32, 64, (2, 2)), nn.ReLU()
        )
        # Compute shape by doing one forward pass
        # with th.no_grad():
        # n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
        # self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        self.apply(init_params)

        assert observation_space.shape[1:] == (7, 7), observation_space.shape
        n = observation_space.shape[1]
        m = observation_space.shape[2]
        self._features_dim = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # return self.linear(self.cnn(observations))
        assert observations.max().item() <= 1.0
        return self.cnn(observations).reshape(-1, self._features_dim)


# https://github.com/facebookresearch/adversarially-motivated-intrinsic-goals/blob/main/monobeast/minigrid/monobeast_amigo.py


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


import easydict
import torch
from torch import nn

flags = easydict.EasyDict(
    goal_dim=10,
    disable_use_embedding=False,
    state_embedding_dim=256,
    num_input_frames=1,  # Frame stack
)

# class MinigridNet(nn.Module):
#     def __init__(self, observation_shape, num_actions, state_embedding_dim=256, num_input_frames=1, use_lstm=False, num_lstm_layers=1):


class MinigridNet(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(MinigridNet, self).__init__(observation_space, features_dim)
        self.observation_shape = observation_space.shape
        # self.num_actions = num_actions
        self.state_embedding_dim = features_dim
        # self.use_lstm = use_lstm
        # self.num_lstm_layers = num_lstm_layers

        self.use_index_select = True
        self.obj_dim = 5
        self.col_dim = 3
        self.con_dim = 2
        self.goal_dim = flags.goal_dim
        self.agent_loc_dim = 10
        self.num_channels = (self.obj_dim + self.col_dim + self.con_dim + 1) * flags.num_input_frames

        # if flags.disable_use_embedding:
        #     print("not_using_embedding")
        #     self.num_channels = (3+1+1+1+1)*num_input_frames

        self.embed_object = nn.Embedding(11, self.obj_dim)
        self.embed_color = nn.Embedding(6, self.col_dim)
        self.embed_contains = nn.Embedding(4, self.con_dim)
        # self.embed_goal = nn.Embedding(self.observation_shape[0]*self.observation_shape[1] + 1, self.goal_dim)
        self.embed_agent_loc = nn.Embedding(
            self.observation_shape[0] * self.observation_shape[1] + 1, self.agent_loc_dim
        )

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu')
        )

        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(in_channels=self.num_channels, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
        )

        self.fc = nn.Sequential(
            init_(nn.Linear(32 + self.obj_dim + self.col_dim, self.state_embedding_dim)),
            nn.ReLU(),
            init_(nn.Linear(self.state_embedding_dim, self.state_embedding_dim)),
            nn.ReLU(),
        )

        # if use_lstm:
        #     self.core = nn.LSTM(self.state_embedding_dim, self.state_embedding_dim, self.num_lstm_layers)

        # init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
        #                        constant_(x, 0))

        # self.policy = init_(nn.Linear(self.state_embedding_dim, self.num_actions))
        # self.baseline = init_(nn.Linear(self.state_embedding_dim, 1))

    # def initial_state(self, batch_size):
    #     """Initializes LSTM."""
    #     if not self.use_lstm:
    #         return tuple()
    #     return tuple(torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size) for _ in range(2))

    def create_embeddings(self, x, id):
        """Generates compositional embeddings."""
        if id == 0:
            objects_emb = self._select(self.embed_object, x[:, :, :, id::3])
        elif id == 1:
            objects_emb = self._select(self.embed_color, x[:, :, :, id::3])
        elif id == 2:
            objects_emb = self._select(self.embed_contains, x[:, :, :, id::3])
        embeddings = torch.flatten(objects_emb, 3, 4)
        return embeddings

    def _select(self, embed, x):
        """Efficient function to get embedding from an index."""
        if self.use_index_select:
            out = embed.weight.index_select(0, x.reshape(-1))
            # handle reshaping x to 1-d and output back to N-d
            return out.reshape(x.shape + (-1, ))
        else:
            return embed(x)

    def agent_loc(self, frames):
        """Returns the location of an agent from an observation."""
        T, B, *_ = frames.shape
        agent_location = torch.flatten(frames, 2, 3)
        agent_location = agent_location[:, :, :, 0]
        agent_location = (agent_location == 10).nonzero()  # select object id
        agent_location = agent_location[:, 2]
        agent_location = agent_location.view(T, B, 1)
        return agent_location

    # def forward(self, inputs, core_state=(), goal=[]):
    def forward(self, observations: th.Tensor) -> th.Tensor:
        """Main Function, takes an observation and a goal and returns and action."""

        # -- [unroll_length x batch_size x height x width x channels]
        # x = inputs["frame"]

        x = observations

        T, B, h, w, *_ = x.shape

        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        # goal = torch.flatten(goal, 0, 1)

        # Creating goal_channel
        # goal_channel = torch.zeros_like(x, requires_grad=False)
        # goal_channel = torch.flatten(goal_channel, 1,2)[:,:,0]
        # for i in range(goal.shape[0]):
        #     goal_channel[i,goal[i]] = 1.0
        # goal_channel = goal_channel.view(T*B, h, w, 1)
        # carried_col = inputs["carried_col"]
        # carried_obj = inputs["carried_obj"]

        # if flags.disable_use_embedding:
        #     x = x.float()
        #     goal = goal.float()
        #     carried_obj = carried_obj.float()
        #     carried_col = carried_col.float()
        # else:
        x = x.long()
        # goal = goal.long()
        # carried_obj = carried_obj.long()
        # carried_col = carried_col.long()
        # -- [B x H x W x K]
        x = torch.cat(
            [
                self.create_embeddings(x, 0),
                self.create_embeddings(x, 1),
                self.create_embeddings(x, 2),
                goal_channel.float()
            ],
            dim=3
        )
        # carried_obj_emb = self._select(self.embed_object, carried_obj)
        # carried_col_emb = self._select(self.embed_color, carried_col)

        # if flags.no_generator:
        #     goal_emb = torch.zeros(goal_emb.shape, dtype=goal_emb.dtype, device=goal_emb.device, requires_grad = False)

        # x = x.transpose(1, 3)
        x = self.feat_extract(x)
        x = x.view(T * B, -1)

        return x

        # carried_obj_emb = carried_obj_emb.view(T * B, -1)
        # carried_col_emb = carried_col_emb.view(T * B, -1)
        # union = torch.cat([x, carried_obj_emb, carried_col_emb], dim=1)
        # core_input = self.fc(union)

        # PZH:
        # core_input = self.fc(x)
        #
        # if self.use_lstm:
        #     core_input = core_input.view(T, B, -1)
        #     core_output_list = []
        #     notdone = (~inputs["done"]).float()
        #     for input, nd in zip(core_input.unbind(), notdone.unbind()):
        #         nd = nd.view(1, -1, 1)
        #         core_state = tuple(nd * s for s in core_state)
        #         output, core_state = self.core(input.unsqueeze(0), core_state)
        #         core_output_list.append(output)
        #     core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        # else:
        #     core_output = core_input
        #     core_state = tuple()

        # policy_logits = self.policy(core_output)
        # baseline = self.baseline(core_output)
        #
        # if self.training:
        #     action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        # else:
        #     action = torch.argmax(policy_logits, dim=1)
        #
        # policy_logits = policy_logits.view(T, B, self.num_actions)
        # baseline = baseline.view(T, B)
        # action = action.view(T, B)

        # return dict(policy_logits=policy_logits, baseline=baseline, action=action), core_state


class FullObsMinigridPolicyNet(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        # def __init__(self, observation_shape, num_actions):
        super(FullObsMinigridPolicyNet, self).__init__(observation_space, features_dim)
        self.observation_shape = observation_space.shape
        # self.num_actions = num_actions

        self.use_index_select = True
        self.obj_dim = 5
        self.col_dim = 3
        self.con_dim = 2
        self.agent_loc_dim = 10
        self.num_channels = (self.obj_dim + self.col_dim + self.con_dim)

        self.embed_object = nn.Embedding(11, self.obj_dim)
        self.embed_color = nn.Embedding(6, self.col_dim)
        self.embed_contains = nn.Embedding(4, self.con_dim)

        # self.embed_agent_loc = nn.Embedding(
        #     num_embeddings=self.observation_shape[0]*self.observation_shape[1] + 1,
        #     embedding_dim=self.agent_loc_dim
        # )
        assert self.observation_shape[1] == self.observation_shape[2]
        self.embed_agent_loc = nn.Embedding(
            num_embeddings=self.observation_shape[1] * self.observation_shape[2] + 1, embedding_dim=self.agent_loc_dim
        )

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu')
        )

        ##Because Fully_observed
        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(in_channels=self.num_channels, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
        )

        # self._features_dim = 32 + self.agent_loc_dim + self.obj_dim + self.col_dim
        self._features_dim = 32 + self.agent_loc_dim

        # self.fc = nn.Sequential(
        #     init_(nn.Linear(32 + self.agent_loc_dim + self.obj_dim + self.col_dim, 1024)),
        #     nn.ReLU(),
        #     init_(nn.Linear(1024, 1024)),
        #     nn.ReLU(),
        # )

        # init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
        #                        constant_(x, 0))

        # self.policy = init_(nn.Linear(1024, self.num_actions))
        # self.baseline = init_(nn.Linear(1024, 1))

    # def initial_state(self, batch_size):
    #     return tuple()

    def _select(self, embed, x):
        if self.use_index_select:
            out = embed.weight.index_select(0, x.reshape(-1))
            # handle reshaping x to 1-d and output back to N-d
            return out.reshape(x.shape + (-1, ))
        else:
            return embed(x)

    def create_embeddings(self, x, id):
        # indices = torch.tensor([i for i in range(x.shape[3]) if i%3==id])
        # object_ids = torch.index_select(x, 3, indices)
        if id == 0:
            # objects_emb = self._select(self.embed_object, x[:,:,:,id::3])
            objects_emb = self._select(self.embed_object, x[:, id::3])
        elif id == 1:
            # objects_emb = self._select(self.embed_color, x[:,:,:,id::3])
            objects_emb = self._select(self.embed_color, x[:, id::3])
        elif id == 2:
            # objects_emb = self._select(self.embed_contains, x[:,:,:,id::3])
            objects_emb = self._select(self.embed_contains, x[:, id::3])
        # embeddings = torch.flatten(objects_emb, 3, 4)  # B, W, H, C, K -> B, W, H, CK
        # embeddings = torch.flatten(objects_emb, 3, 4)
        assert objects_emb.shape[1] == 1
        embeddings = torch.squeeze(objects_emb, dim=1)
        return embeddings

    def agent_loc(self, frames):
        # T, B, *_ = frames.shape
        # frames = T, B, H, W, C
        # agent_location = torch.flatten(frames, 2, 3)  # Should be: T, B, WxH, C
        # agent_location = agent_location[:,:,:,0]  # T, B, WxH
        # agent_location = (agent_location == 10).nonzero() #select object id (3D index)
        # agent_location = agent_location[:,2]  # The location in (WxH)
        # agent_location = agent_location.view(T,B,1)

        agent_location = torch.flatten(frames, 2, 3)  # Should be: B, C, H, W -> B, C, WH
        agent_location = agent_location[:, 0]  # B, C, WH -> B, WH
        agent_location = (agent_location == 10).nonzero()  # select object id (2D index)
        agent_location = agent_location[:, 1]  # -> B, 1
        agent_location = agent_location.view(-1, 1)

        return agent_location

    def forward(self, obs):
        # -- [unroll_length x batch_size x height x width x channels]
        x = obs
        # T, B, *_ = x.shape

        # -- [unroll_length*batch_size x height x width x channels]
        # x = torch.flatten(x, 0, 1)  # Merge time and batch.
        agent_loc = self.agent_loc(obs)
        # carried_col = inputs["carried_col"]
        # carried_obj = inputs["carried_obj"]

        x = x.long()
        agent_loc = agent_loc.long()
        # carried_obj = carried_obj.long()
        # carried_col = carried_col.long()
        # -- [B x H x W x K]
        x = torch.cat([self.create_embeddings(x, 0), self.create_embeddings(x, 1), self.create_embeddings(x, 2)], dim=3)
        agent_loc_emb = self._select(self.embed_agent_loc, agent_loc)
        # carried_obj_emb = self._select(self.embed_object, carried_obj)
        # carried_col_emb = self._select(self.embed_color, carried_col)

        # -- [unroll_length*batch_size x channels x width x height]
        x = x.transpose(1, 3)
        # -- [B x K x W x H]
        # PZH: correct! x is now [B, C, W, H] (though the C is reconstructed channels!)

        # agent_loc_emb = agent_loc_emb.view(T * B, -1)
        assert agent_loc_emb.shape[1] == 1
        agent_loc_emb = torch.squeeze(agent_loc_emb, dim=1)

        # carried_obj_emb = carried_obj_emb.view(T * B, -1)
        # carried_col_emb = carried_col_emb.view(T * B, -1)

        x = self.feat_extract(x)
        # x = x.view(T * B, -1)
        x = x.view(x.shape[0], -1)
        # union = torch.cat([x, agent_loc_emb, carried_obj_emb, carried_col_emb], dim=1)
        union = torch.cat([x, agent_loc_emb], dim=1)

        return union
        # core_input = self.fc(union)

        # core_output = core_input
        # core_state = tuple()

        # policy_logits = self.policy(core_output)
        # baseline = self.baseline(core_output)
        #
        #
        # if self.training:
        #     action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        # else:
        #     # Don't sample when testing.
        #     action = torch.argmax(policy_logits, dim=1)
        #
        # policy_logits = policy_logits.view(T, B, self.num_actions)
        # baseline = baseline.view(T, B)
        # action = action.view(T, B)
        #
        # return dict(policy_logits=policy_logits, baseline=baseline, action=action), core_state
