import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

class ActorCritic(nn.Module):
	def __init__(self, state_dim, action_dim, has_continuous_action_space,
				 action_std_init, device = torch.device('cpu')):
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
