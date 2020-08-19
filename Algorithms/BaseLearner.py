import numpy as np

from Problems.BaseProblem import BaseProblem


class BaseLearner:
    def __init__(self, problem: BaseProblem, **kwargs):
        self.problem = problem
        self.w = np.zeros(kwargs['num_features'])
        self.gamma = kwargs['gamma']
        self.alpha = kwargs['alpha']
        self.lmbda = kwargs['lmbda']
        self.feature_rep = np.load('feature_rep.npy')[:, :, kwargs['run']]
        self.state = -1

    def compute_step_size(self, transition):
        return self.alpha

    def choose_behavior_action(self):
        return self.problem.select_behavior_action(self.feature_rep[self.state, :])

    def choose_target_action(self):
        return self.problem.select_target_action(self.feature_rep[self.state, :])

    def learn(self, *args):
        raise NotImplementedError
