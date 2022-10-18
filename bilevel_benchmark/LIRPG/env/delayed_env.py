import gym
import torch
import numpy as np
from env.point_mass import ContextualPointMass


class DelayedEnv(gym.Wrapper):

    def __init__(self, env_id,  use_intrinsic_reward=False, reward_freq=20, mixing_wieght=1):

        if env_id == 'PointMass':
            self.env = ContextualPointMass(use_intrinsic_reward=False)
        else:
            self.env = gym.make(env_id)

        super().__init__(self.env)

        self.delayed_reward = 0
        self.reward_freq = reward_freq
        self.mixing_wieght = mixing_wieght

        self.t = 0
        self.step_counter = 0
        self.state = self.env.reset()
        self._max_episode_steps = self.env._max_episode_steps

        self.use_intrinsic_reward = use_intrinsic_reward
        self.intrinsic_reward = None

    def set_intrinsic_reward(self, intrinsic_reward):
        self.intrinsic_reward = intrinsic_reward

    def step(self, action):

        observation, gt_ex_reward, done, _ = super().step(action)
        self.delayed_reward += gt_ex_reward
        self.state = observation
        self.t += 1
        self.step_counter +=1

        if (self.t+1) % self.reward_freq == 0 \
                or done:  # or done?
            ex_reward = self.delayed_reward
            self.delayed_reward = 0.
            self.step_counter = 0
        else:
            ex_reward = 0

        # additive intrinsic reward; from in_reward_network
        if self.use_intrinsic_reward:
            in_reward = torch.squeeze(self.intrinsic_reward.forward(np.append(self.state, action)))  # req grad = true
            add_reward = self.mixing_wieght * ex_reward + in_reward  # TODO
        else:
            in_reward = 0
            add_reward = ex_reward

        info = {"in_reward": in_reward,
                "ex_reward": ex_reward}# TODO


        return observation, add_reward, done, info

    def reset(self):
        self.delayed_reward = 0
        self.t = 0
        return super().reset()

