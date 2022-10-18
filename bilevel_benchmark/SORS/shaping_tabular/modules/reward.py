import numpy as np


def distance(gt_reward, ter1, ter2, gamma):
    R1 = calculate_return(gt_reward, ter=ter1, gamma=gamma)
    R2 = calculate_return(gt_reward, ter=ter2, gamma=gamma)
    return R1 >= R2


def calculate_return(reward_function, ter, gamma):  # ter = [[si], [ai], [ri]]
    _return = 0
    for t in range(len(ter[1])):
        s = ter[0][t]
        a = ter[1][t]
        _return += gamma ** t * reward_function[s][a]
    return _return


def calculate_p_fucntion(reward, ter1, ter2, gamma):  # P(T1>T2)
    R1 = calculate_return(reward, ter1, gamma)
    R2 = calculate_return(reward, ter2, gamma)
    return np.exp(R1) / (np.exp(R1)+np.exp(R2))


def calculate_trejecotry_gradiant(ter, state, action, gamma):
    g = 0
    for t in range(len(ter[1])):
        if ter[0][t] == state and ter[1][t] == action:
            g += gamma ** t * 1

    return g


def calculate_gradiant(reward_function, n_states, n_actions, ter1, ter2, gamma):  # ter1 >rs ter2
    gradiant = np.zeros_like(reward_function)
    p = calculate_p_fucntion(reward_function, ter1, ter2, gamma)
    for state in range(n_states):
        for action in range(n_actions):
            g1 = calculate_trejecotry_gradiant(ter1, state, action, gamma)
            g2 = calculate_trejecotry_gradiant(ter2, state, action, gamma)
            gradiant[state][action] += (1-p) * (g2 - g1)  # p * (1/p-1) * g

    return gradiant


class reward():
    def __init__(self, gt_reward, n_states=5, n_actions=2):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gt_reward = gt_reward
        self.reward_function = np.zeros(shape=(n_states, n_actions), dtype=float)

    def get_reward_function(self):
        return self.reward_function

    def update(self, trj_batch,
               learning_rate=0.1, gamma=0.9):
        """
        updating the reward parameter,
        using batch of trajectories.
        """
        for i in range(len(trj_batch)):  # first trajectory:Ti
            for j in range(len(trj_batch)):  # second trajectory:Tj
                d = distance(self.gt_reward, trj_batch[i], trj_batch[j], gamma=gamma)
                gradiant = 0
                if d:  # Ti >rs Tj
                    gradiant = calculate_gradiant(self.reward_function, self.n_states, self.n_actions,
                                                  trj_batch[i], trj_batch[j], gamma=gamma)  # Ti > Tj
                else:  # Tj >rs To
                    gradiant = calculate_gradiant(self.reward_function, self.n_states, self.n_actions,
                                                  trj_batch[j], trj_batch[i], gamma=gamma)  # Ti < Tj

                self.reward_function -= learning_rate * gradiant  # here?

