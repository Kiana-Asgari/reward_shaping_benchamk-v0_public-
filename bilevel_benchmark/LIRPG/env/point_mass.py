from abc import ABC

import numpy as np
import torch
import time

from gym import Env, spaces


class ContextualPointMass(Env, ABC):

    def __init__(self, epsilon=20, context=np.array([0., 2., 2.0]), use_intrinsic_reward=False):

        self.action_space = spaces.Box(np.array([-10., -10.]), np.array([10., 10.]))
        self.observation_space = spaces.Box(np.array([-4., -np.inf, -4., -np.inf]),
                                            np.array([4., np.inf, 4., np.inf]))

        self._state = None
        self._goal_state = np.array([0., 0., -3., 0.])
        self.context = context
        self._dt = 0.01
        self._max_episode_steps = 80
        self.time_step = 0
        self.epsilon = epsilon

        self.use_intrinsic_reward = use_intrinsic_reward
        self.intrinsic_reward = None

    def set_intrinsic_reward(self, intrinsic_reward):
        self.intrinsic_reward = intrinsic_reward

    def reset(self, **kwargs):
        self._state = np.array([0., 0., 3., 0.])
        self.time_step = 0
        return np.copy(self._state)

    def _step_internal(self, state, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        state_der = np.zeros(4)
        state_der[0::2] = state[1::2]
        friction_param = self.context[2]
        state_der[1::2] = 1.5 * action - friction_param * state[1::2] + np.random.normal(0, 0.05, (2,))
        new_state = np.clip(state + self._dt * state_der, self.observation_space.low,
                            self.observation_space.high)

        crash = False
        if state[2] >= 0 > new_state[2] or state[2] <= 0 < new_state[2]:
            alpha = (0. - state[2]) / (new_state[2] - state[2])
            x_crit = alpha * new_state[0] + (1 - alpha) * state[0]

            if np.abs(x_crit - self.context[0]) > 0.5 * self.context[1]:
                new_state = np.array([x_crit, 0., 0., 0.])
                crash = True

        return new_state, crash

    def step(self, action):
        if self._state is None:
            raise RuntimeError("State is None! Be sure to reset the environment before using it")
        self.time_step += 1

        action = np.clip(action, self.action_space.low, self.action_space.high)

        new_state = self._state
        crash = False
        for i in range(0, 10):
            new_state, crash = self._step_internal(new_state, action)
            if crash:
                break

        if self.time_step == self._max_episode_steps:
            crash = True

        # extrinsic reward; from original sparse environment
        if np.linalg.norm(self._goal_state[0::2] - new_state[0::2]) < self.epsilon:  # TODO
            ex_reward = np.exp(-0.6 * np.linalg.norm(self._goal_state[0::2] - new_state[0::2]))  # TODO small reward
        else:
            ex_reward = 0

        # additive intrinsic reward; from in_reward_network
        if self.use_intrinsic_reward:
            in_reward = torch.squeeze(self.intrinsic_reward.forward(np.append(self._state, action)).detach()).item()
            reward = ex_reward + in_reward  # TODO
        else:
            in_reward = 0
            reward = ex_reward

        self._state = np.copy(new_state)

        info = {"in_reward": in_reward,
                "ex_reward": ex_reward}

        return new_state, reward, crash, info

    @property
    def max_episode_steps(self):
        return self._max_episode_steps
