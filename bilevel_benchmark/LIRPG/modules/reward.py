import math

import torch
import numpy as np
import torch.nn as nn

from .policies import MlpPolicyIntrinsicReward, MlpPolicyExValue
from utils import assign_grads


# Continuous Reward Network
class IntrinsicReward:
    def __init__(self, obs_shape, action_shape, args):

        self.obs_shape = obs_shape
        self.action_shape = action_shape

        self.gamma = args.gamma
        self.eps_clip = args.eps_clip

        self.in_reward_model = MlpPolicyIntrinsicReward(dim_action=action_shape, dim_obs=obs_shape).to(args.device)

        self.ex_value_model = MlpPolicyExValue(dim_action=action_shape, dim_obs=obs_shape).to(args.device)

        self.optimizer = torch.optim.Adam([
            {'params': self.in_reward_model.parameters(), 'lr': args.lr_in_reward},
            {'params': self.ex_value_model.parameters(), 'lr': args.lr_ex_value}
        ])

        self.MseLoss = nn.MSELoss()
        self.device = args.device

    def update(self, buffer, K_epochs, policy_model):
        # Monte Carlo estimate of returns G_ex
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(buffer.ex_rewards), reversed(buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the extrinsic rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(buffer.actions, dim=0)).detach().to(self.device)
        old_states_actions = torch.squeeze(torch.stack(buffer.states_actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(buffer.logprobs, dim=0)).detach().to(self.device)

        # Optimize for K epochs
        for _ in range(K_epochs):
            # Evaluating old actions and values
            logprobs = policy_model.evaluate(old_states, old_actions)

            v_ex = self.ex_value_model.forward(old_states_actions)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # match state_values tensor dimensions with rewards tensor
            v_ex = torch.squeeze(v_ex)

            # Normalizing the extrinsic advantages
            advantages = rewards - v_ex.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Final loss of clipped objective
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(v_ex, rewards)

            # ToDo : batches
            # take gradient step
            self.optimizer.zero_grad(set_to_none=True)

            assign_grads(loss.mean(), self.in_reward_model.network, create_graph=False)
            assign_grads(loss.mean(), self.ex_value_model.network, create_graph=False)

            self.optimizer.step()

    def get_intrinsic_reward_network(self):
        return self.in_reward_model
