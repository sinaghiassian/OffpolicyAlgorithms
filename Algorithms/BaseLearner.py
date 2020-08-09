import numpy as np


class BaseLearner:
    def __init__(self, parameters):
        self.w = np.zeros(parameters['feature_size'])
        self.gamma = parameters['gamma']
        self.alpha = parameters['alpha']

    def compute_step_size(self, transition):
        return self.alpha

    def learn(self, *args):
        raise NotImplementedError
