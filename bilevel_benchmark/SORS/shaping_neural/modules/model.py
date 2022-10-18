import torch
from torch import nn


class DiscreteRewardNetwork(nn.Module):
    # contains neural network
    def __init__(self, n_inputs, n_outputs, n_hidden=16):
        super(DiscreteRewardNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.Tanh(),
            nn.utils.weight_norm(nn.Linear(n_hidden, n_outputs))
        )

    def forward(self, state):
        rewards = self.network(torch.FloatTensor(state))
        return rewards


class ContinuousRewardNetwork(nn.Module):
    # contains neural network
    def __init__(self, dim_action, dim_obs):
        super(ContinuousRewardNetwork, self).__init__()

        self.network = nn.Sequential(
                nn.Linear(dim_obs+dim_action, 64),
                nn.Tanh(),
                nn.Linear(64, 256),
                nn.Tanh(),
                nn.Linear(256, 64),
                nn.Tanh(),
                nn.Linear(64, 16),
                nn.Tanh(),
                nn.utils.weight_norm(nn.Linear(16, 1))
        )
        #self.network.apply(self.init_weights)

    def init_weights(layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform(layer.weight)
            layer.bias.data.fill_(0.01)

    def forward(self, state_actoin):
        rewards = self.network(torch.FloatTensor(state_actoin))
        return rewards
