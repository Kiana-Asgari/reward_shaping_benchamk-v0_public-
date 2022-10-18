import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical


############################
#      Reward Networks     #
############################
class MlpPolicyIntrinsicReward(nn.Module):
    def __init__(self, dim_obs, dim_action):
        super(MlpPolicyIntrinsicReward, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(dim_obs + dim_action, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

        # self.network.apply(self.init_weights)

    def forward(self, state_action):
        reward = self.network(torch.FloatTensor(state_action))
        return reward


class MlpPolicyExValue(nn.Module):
    def __init__(self, dim_obs, dim_action):
        super(MlpPolicyExValue, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(dim_obs + dim_action, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        # self.network.apply(self.init_weights)

    def forward(self, state_action):
        value = self.network(torch.FloatTensor(state_action))
        return value


############################
#      Policy Networks     #
############################
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space,
                 action_std_init, device=torch.device('cpu')):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.device = device

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim))
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1))

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1))

    def forward(self):
        raise NotImplementedError

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def act(self, state):  # running actor to predict the next action
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):  # running crititc to get the estimated value function

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class NewPolicy(nn.Module):
    def __init__(self, state_dim, action_dim,
                 action_std_init,
                 layer1_weights, layer2_weights, layer3_weights,
                 layer1_bias, layer2_bias, layer3_bias,
                 device=torch.device('cpu'), has_continuous_action_space=True):

        super(NewPolicy, self).__init__()

        if has_continuous_action_space:
            self.l1 = nn.Linear(state_dim, 64)
            self.l2 = nn.Linear(64, 64)
            self.l3 = nn.Linear(64, action_dim)
        else:
            self.l1 = nn.Linear(state_dim, 64)
            self.l2 = nn.Linear(64, 64)
            self.l3 = nn.Linear(64, action_dim)

        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        self.device = device
        self.action_dim = action_dim
        self.has_continuous_action_space = has_continuous_action_space



        del self.l1.weight
        del self.l2.weight
        del self.l3.weight

        self.l1.weight = layer1_weights
        self.l2.weight = layer2_weights
        self.l3.weight = layer3_weights

        del self.l1.bias
        del self.l2.bias
        del self.l3.bias

        self.l1.bias = layer1_bias
        self.l2.bias = layer2_bias
        self.l3.bias = layer3_bias

    def evaluate(self, state, action):

        action_mean = self.forward(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)

        # For Single Action Environments.
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)

        action_logprobs = dist.log_prob(action)

        return action_logprobs

    def forward(self, state):
        action = torch.tanh(self.l1(state))
        action = torch.tanh(self.l2(action))
        action = self.l3(action)

        if not self.has_continuous_action_space:
            action = torch.softmax(action, dim=-1)

        return action
