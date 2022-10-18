import gym
import numpy as np
from env.point_mass import ContextualPointMass


class DelayedEnv(gym.Wrapper):

    def __init__(self, env_id, use_dynamic_reward_function, is_continuous=False, reward_network=None, reward_freq=20):

        if env_id == 'PointMass':
            self.env = ContextualPointMass()
        else:
            self.env = gym.make(env_id)

        super().__init__(self.env)

        self.delayed_reward = 0
        self.reward_freq = reward_freq
        self.use_dynamic_reward_function = use_dynamic_reward_function
        self.is_continuous = is_continuous
        self.reward_network = reward_network
        self.t = 0
        self.step_counter = 0
        self.state = self.env.reset()
        self._max_episode_steps = self.env.max_episode_steps

    def set_reward(self, reward_network):

        if self.use_dynamic_reward_function:
            self.reward_network = reward_network

        else:
            print('---------- Warning: setting dynamic reward for non-dynamic environment ------------')

    def step(self, action):

        observation, orig_reward, done, info = super().step(action)
        self.delayed_reward += orig_reward * (0.9**(self.step_counter-self.reward_freq+1))
        self.state = observation
        self.t += 1
        self.step_counter +=1

        if self.use_dynamic_reward_function and self.is_continuous:
            reward_vector = self.reward_network.forward(np.append(self.state, action)).detach().numpy()
            reward = reward_vector[0]

        elif self.use_dynamic_reward_function:
            reward_vector = self.reward_network.forward(self.state).detach().numpy()
            reward = reward_vector[action]

        else:
            if (self.t+1) % self.reward_freq ==0 or self.max_episode_steps == self.time_step:  # or done?
                reward = self.delayed_reward
                self.delayed_reward = 0.
                self.step_counter = 0
            else:
                reward = 0

        return observation, reward, done, info

    def reset(self):
        self.delayed_reward = 0
        self.t = 0
        return super().reset()

