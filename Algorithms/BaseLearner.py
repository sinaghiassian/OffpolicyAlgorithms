import numpy as np


class BaseLearner:
    def __init__(self, **kwargs):
        self.w = np.zeros(kwargs['feature_size'])
        self.gamma = kwargs['gamma']
        self.alpha = kwargs['alpha']

    def compute_step_size(self, transition):
        return self.alpha

    def learn(self, *args):
        raise NotImplementedError
