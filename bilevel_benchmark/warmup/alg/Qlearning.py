"""
Tabular Q learning algorithm
"""

class Qlearning():
    def __init__(self, env, test_env):
        self.env = env
        self.test_env = test_env
        self.n_action = self.env.action_space.n
        self.n_state = self.env.observation_space.n


    def forward(self):
        pass

    def backward(self):
        pass


    def learn(self, total_episodes, print_interval,
              learning_rate, gamma, epsilon):
        pass