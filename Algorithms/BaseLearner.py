import numpy as np


class BaseLearner:
    def __init__(self, **kwargs):
        self.w = np.zeros(kwargs['num_features'])
        self.gamma = kwargs['gamma']
        self.alpha = kwargs['alpha']
        self.lmbda = kwargs['lmbda']

    def compute_step_size(self, transition):
        return self.alpha

    def learn(self, *args):
        raise NotImplementedError

    def choose_behavior_action(self):
        raise NotImplementedError

    def choose_target_action(self):
        raise NotImplementedError
