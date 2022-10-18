"""
Delayed Chain environment,
start at the state 1


Action space = {(False)0:->, (True)1:<-}
Agent slips with probability 0.2
Agent will reach the terminal state from leftmost and rightmost states
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class DelayedChainEnv(gym.Env):

    def __init__(self, n_states=5, n_actions=2):
        self.n_states = n_states
        self.n_actions = n_actions
        self.reward = np.zeros(shape=(n_states, n_actions))  # parametrized reward function
        self.reward[n_states - 1][0] = 100
        self.state = 1  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(n_states)

        self.n_steps = 0
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_reward(self, reward):
        self.reward = reward

    def get_reward(self):
        return self.reward

    def step(self, action):
        assert self.action_space.contains(action)
        self.n_steps += 1

        _reward = self.reward[self.state, action]
        done = self.n_steps > 200
        # next state:
        if action and self.state > 0:  # 'backwards'
            self.state = self.state - 1

        elif action and self.state == 0:  # 'backwards': reaching the Terminal state
            done = True

        elif self.state < self.n_states - 1:  # 'forwards'
            self.state += 1

        else:  # 'forwards': reaching the Terminal state
            done = True

        return self.state, _reward, done, {}

    def reset(self):
        self.state = 1
        self.n_steps = 0
        return self.state
