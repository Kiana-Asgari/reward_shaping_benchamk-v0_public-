import numpy as np
import time


from gym import Env, spaces

class ContextualPointMass(Env):

    def __init__(self, context=np.array([0., 2., 2.0])):
        self.action_space = spaces.Box(np.array([-10., -10.]), np.array([10., 10.]))
        self.observation_space = spaces.Box(np.array([-4., -np.inf, -4., -np.inf]),
                                            np.array([4., np.inf, 4., np.inf]))

        self._state = None
        self._goal_state = np.array([0., 0., -3., 0.])
        self.context = context
        self._dt = 0.01
        self.max_episode_steps = 80
        self.time_step = 0

    def reset(self):
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

        self._state = np.copy(new_state)

        info = {"success": np.linalg.norm(self._goal_state[0::2] - new_state[0::2]) < 0.25}

        #print(info)
        #print('      ',crash, np.linalg.norm(self._goal_state[0::2] - new_state[0::2]))

        if self.time_step == self.max_episode_steps:
            crash = True

        return new_state, np.exp(-0.6 * np.linalg.norm(self._goal_state[0::2] - new_state[0::2])), crash, info


