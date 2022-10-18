import math

import torch
import numpy as np
from modules.model import DiscreteRewardNetwork, ContinuousRewardNetwork
from utils import distance, p_function


class DiscreteReward:
    def __init__(self, obs_shape, n_actions, args):
        self.obs_shape = obs_shape
        self.n_actions = n_actions

        self.gamma = args.gamma

        self.reward_network = DiscreteRewardNetwork(n_inputs=obs_shape, n_outputs=n_actions)
        self.reward_optimizer = torch.optim.Adam(self.reward_network.parameters(), lr=args.lr_reward)

    def get_reward_network(self): return self.reward_network

    def loss(self, trj1, trj2):

        if distance(ter1=trj1, ter2=trj2, gamma=self.gamma) > 0:  # I(T1 <gt T2)
            state1_tensor = torch.FloatTensor(np.array(trj1[0]))
            state2_tensor = torch.FloatTensor(np.array(trj2[0]))
            action1_tensor = torch.LongTensor(np.array(trj1[1]))
            action2_tensor = torch.LongTensor(np.array(trj2[1]))

        else:  # 1 - I(T1 <gt T2)
            state1_tensor = torch.FloatTensor(np.array(trj2[0]))
            state2_tensor = torch.FloatTensor(np.array(trj1[0]))
            action1_tensor = torch.LongTensor(np.array(trj2[1]))
            action2_tensor = torch.LongTensor(np.array(trj1[1]))

        reward_vector1 = self.reward_network.forward(state1_tensor)
        reward_vector2 = self.reward_network.forward(state2_tensor)

        selected_reward1 = torch.gather(reward_vector1, 1, action1_tensor.unsqueeze(1)).squeeze()
        selected_reward2 = torch.gather(reward_vector2, 1, action2_tensor.unsqueeze(1)).squeeze()
        loss = -1 * p_function(selected_reward1, selected_reward2, self.gamma)

        return loss

    def update(self, trj_batch):
        # Updating reward parameters on a trajectory batch
        # trj_batch: shape(batch_size, [3]episode_length)
        loss = 0
        for i, trj1 in enumerate(trj_batch):
            for j, trj2 in enumerate(trj_batch):
                loss += self.loss(trj1, trj2)

        # loss /= len(trj_batch) # Normalizing loss?
        self.reward_optimizer.zero_grad()
        loss.backward()
        self.reward_optimizer.step()


# Continuous Reward Network
class ContinuousReward:
    def __init__(self, obs_shape, action_shape, args):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.gamma = args.gamma

        self.reward_network = ContinuousRewardNetwork(dim_action=action_shape, dim_obs=obs_shape)
        self.reward_optimizer = torch.optim.Adam(self.reward_network.parameters(), lr=args.reward_lr)

    def get_reward_network(self): return self.reward_network

    def loss(self, trj1, trj2):
        if distance(ter1=trj1, ter2=trj2, gamma=self.gamma) > 0:  # I(T1 <gt T2)
            transition1_tensor = torch.FloatTensor(np.array(trj1[3]))
            transition2_tensor = torch.FloatTensor(np.array(trj2[3]))

        else:  # 1 - I(T1 <gt T2)
            transition1_tensor = torch.FloatTensor(np.array(trj2[3]))
            transition2_tensor = torch.FloatTensor(np.array(trj1[3]))

        if math.isnan(torch.tensor(self.reward_network.network[0].weight)[0][0]):
            print('network is nan befor forwarding')

        reward1 = self.reward_network.forward(transition1_tensor).squeeze()
        _temp = reward1[1]  # TODO
        reward2 = self.reward_network.forward(transition2_tensor).squeeze()
        _temp2 = reward2[1]

        if(math.isnan(_temp.item())):
            print('forward is nan')
        loss = -1 * p_function(reward1, reward2, self.gamma)

        return loss

    def update(self, trj_batch):
        # Updating reward parameters on a trajectory batch
        # trj_batch: shape(batch_size, [3]episode_length)
        loss = 0
        for i, trj1 in enumerate(trj_batch):
            for j, trj2 in enumerate(trj_batch):
                temp = self.loss(trj1, trj2)
                if math.isnan(temp):
                    print('nan in loss funciton')
                loss += temp


        # loss /= len(trj_batch) # Normalizing loss?

        if math.isnan(torch.tensor(self.reward_network.network[0].weight)[0][0]):
            print('nan befor zero grad')
        self.reward_optimizer.zero_grad()
        if math.isnan(torch.tensor(self.reward_network.network[0].weight)[0][0]):
            print('nan after zero grad')
        loss.backward()
        if math.isnan(torch.tensor(self.reward_network.network[0].weight)[0][0]):
            print('nan after backward')
        self.reward_optimizer.step()
        if math.isnan(torch.tensor(self.reward_network.network[0].weight)[0][0]):
            print('nan after step')


    # Temporary function
    def test(self, buffer):
        loss = 0
        for i, trj1 in enumerate(buffer):
            for j, trj2 in enumerate(buffer):
                if distance(ter1=trj1, ter2=trj2, gamma=self.gamma) > 0:  # I(T1 <gt T2)
                    transition1_tensor = torch.FloatTensor(np.array(trj1[3]))
                    transition2_tensor = torch.FloatTensor(np.array(trj2[3]))

                else:  # 1 - I(T1 <gt T2)
                    transition1_tensor = torch.FloatTensor(np.array(trj2[3]))
                    transition2_tensor = torch.FloatTensor(np.array(trj1[3]))

                reward1 = self.reward_network.forward(transition1_tensor).squeeze().detach().numpy()
                reward2 = self.reward_network.forward(transition2_tensor).squeeze().detach().numpy()

                R1 = np.array([0.9 ** i * reward1[i] for i in range(len(reward1))]).sum()
                R2 = np.array([0.9 ** i * reward2[i] for i in range(len(reward2))]).sum()

                if R1 < R2 and i > j:  #e?
                    loss += 1
        print('|            Reward Mismatch: {}            '.format(loss))
        if loss == 0:
            pass
        return loss





