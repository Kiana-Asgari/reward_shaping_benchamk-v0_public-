import numpy as np
import matplotlib.pyplot as plt


"""
TABULAR REINFORCE ALGORITHM
"""

def calculate_return(reward_buffer, t, gamma):
    discount = np.power(gamma, range(len(reward_buffer) - t-1))
    return np.inner(discount, reward_buffer[t + 1:])


def calculate_gradiant(state, action, parameter_policy):
    gradiant = np.zeros_like(parameter_policy)
    gradiant[state] -= np.exp(parameter_policy[state])\
                              / np.sum(np.exp(parameter_policy[state]), axis=0)
    gradiant[state][action] += 1
    return gradiant

def plot_learning_curve(learning_reward, log_dir, name):
    plt.figure()
    plt.plot(range(len(learning_reward)), learning_reward)
    plt.xlabel('episode')
    plt.ylabel('Episode return')
    plt.title(name)
    #plt.show()
    plt.savefig(log_dir + name+'.png')


class REINFORCE():
    def __init__(self, env, test_env):
        self.env = env
        self.test_env = test_env
        self.n_action = self.env.action_space.n
        self.n_state = self.env.observation_space.n
        self.parameter_policy = np.zeros(shape=(self.n_state, self.n_action), dtype=float)
        self.test_rewards = []

    def forward(self, state):
        # softmax layer
        prob = np.exp(self.parameter_policy[state]) / np.sum(np.exp(self.parameter_policy[state]), axis=0)
        return np.random.choice(range(self.n_action), p=prob)

    def backward(self, reward_buffer, state_buffer, action_buffer, learning_rate, gamma):
        # policy update
        for t in range(len(reward_buffer) - 1):
            G = calculate_return(reward_buffer, t, gamma)
            gradiant = calculate_gradiant(state_buffer[t], action_buffer[t], self.parameter_policy)
            self.parameter_policy += learning_rate * (gamma ** t) * G * gradiant

    def run_single_episode(self, env, reward_buffer=None, state_buffer=None, action_buffer=None):
        # saving single trajectory
        state = env.reset()
        done = False
        state_buffer.append(state)
        reward_buffer.append(0)

        while not done:
            action = self.forward(state)
            action_buffer.append(action)

            state, reward, done, _ = env.step(action)
            state_buffer.append(state)
            reward_buffer.append(reward)
        return state_buffer, action_buffer, reward_buffer



    def learn(self, total_episodes, log_interval, log_dir, log_num, name,
              learning_rate, gamma):
        for n_epi in range(total_episodes):
            reward_buffer = []
            state_buffer = []
            action_buffer = []
            self.run_single_episode(self.env, reward_buffer, state_buffer, action_buffer)
            print('[episode]', n_epi)
            print('     states: ', state_buffer)
            self.backward(reward_buffer, state_buffer, action_buffer,
                          learning_rate, gamma)
            if n_epi % log_interval == 0:
                self.test(gamma, log_num)

        plot_learning_curve(self.test_rewards, log_dir, name=name)


    def test(self, gamma, log_num):
        _return = 0
        for i in range(log_num):
            states, actions, rewards = self.run_single_episode(env = self.test_env, reward_buffer=[],
                                                           state_buffer=[],
                                                           action_buffer=[])
            _return += calculate_return(rewards, t=0, gamma=gamma)
        self.test_rewards.append(_return/log_num)




