import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(learning_reward, log_dir, name):
    plt.figure()
    plt.plot(range(len(learning_reward)), learning_reward)
    plt.xlabel('episode')
    plt.ylabel('Episode return')
    plt.title(name)
    #plt.show()
    plt.savefig(log_dir + name+'.png')


def discount_rewards(rewards, gamma=0.9):
    #G = 0
    #G_array = np.zeros(len(rewards))
    #for t, i in enumerate(range(len(rewards)-1, -1, -1)):

    #    G = rewards[i] + (gamma ** t) * G
    #    G_array[i] = G

    G_old = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    G_old = G_old[::-1].cumsum()[::-1]

    # print(G_old - G_array)
    return G_old


class policy_network():
    # contains neural network
    def __init__(self, env, n_hidden):
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n

        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, self.n_outputs),
            nn.Softmax(dim=-1)  # squashes values between 0 and 1
        )

    def forward(self, state):
        action_probs = self.network(torch.FloatTensor(state))
        return action_probs


class n_Reinforce():
    def __init__(self, env, test_env, policy_network, optimizer):
        self.env = env
        self.test_env = test_env
        self.policy_network = policy_network
        self.optimizer = optimizer
        self.test_rewards = []

    def loss(self, states, actions, rewards):
        logpolicy = torch.log(self.policy_network.forward(states))
        # print('prob ',self.policy_network.forward(states))
        selected_logpolicy = rewards * torch.gather(logpolicy, 1, actions.unsqueeze(1)).squeeze()
        loss = - selected_logpolicy.mean()
        return loss

    def run_single_episode(self, env, action_space):
        s_0 = env.reset()

        states = []
        rewards = []
        actions = []
        done = False

        while not done:
            action_probs = self.policy_network.forward(state=s_0).detach().numpy()
            action = np.random.choice(action_space, p=action_probs)
            s_1, r, done, _ = self.env.step(action)

            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1

        return states, actions, rewards

    def learn(self, num_episodes, batch_size=10, gamma=0.99,
              log_interval=32, log_num=16, log_dir='', name=''):

        learning_rewards = []

        batch_rewards = []
        batch_actions = []
        batch_states = []
        bc = 0

        action_space = np.arange(self.env.action_space.n)

        for ep in range(num_episodes):
            states, actions, rewards_array = self.run_single_episode(self.env, action_space)

            # episode is done
            batch_rewards.extend(discount_rewards(rewards_array, gamma))

            batch_states.extend(states)
            batch_actions.extend(actions)
            print('ep', ep, 'rew', sum(rewards_array))
            bc += 1

            # batch is completed, updating the network
            if bc == batch_size:
                state_tensor = torch.FloatTensor(np.array(batch_states))
                reward_tensor = torch.FloatTensor(np.array(batch_rewards))
                action_tensor = torch.LongTensor(np.array(batch_actions))

                loss = self.loss(state_tensor, action_tensor, reward_tensor)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()



                batch_rewards = []  # new batch started
                batch_actions = []
                batch_states = []
                bc = 0

            if ep % log_interval == 0:
                self.test(gamma, log_num)
        print(self.test_rewards)
        plot_learning_curve(self.test_rewards, log_dir, name=name)
        #plot_learning_curve(learning_rewards, log_dir, name=name)



    def test(self, gamma, log_num):
        _return = 0
        action_space = np.arange(self.env.action_space.n)

        for i in range(log_num):
            states, actions, rewards = self.run_single_episode(env=self.env, action_space=action_space)
            _return += sum(rewards)

        self.test_rewards.append(_return / log_num)