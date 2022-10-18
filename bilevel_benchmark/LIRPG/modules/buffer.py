import numpy as np

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.states_actions = []

        self.logprobs = []

        self.ex_rewards = []
        self.inex_rewards = []

        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.states_actions[:]

        del self.logprobs[:]

        del self.ex_rewards[:]
        del self.inex_rewards[:]

        del self.is_terminals[:]

