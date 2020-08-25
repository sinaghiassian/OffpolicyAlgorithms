from abc import ABC

import numpy as np

from Problems.BaseProblem import BaseProblem


class BaseAgent:
    def __init__(self, problem: BaseProblem, **kwargs):
        self.problem = problem
        self.w = np.zeros(self.problem.num_features)
        self.gamma = kwargs['GAMMA']
        self.alpha = kwargs['alpha']
        self.lmbda = kwargs['lmbda']
        self.state_values = self.problem.get_state_value
        self.d_mu = self.problem.get_behavior_dist
        self.state = -1
        self.next_state = -1
        self.action = -1

    def compute_rmsve(self):
        est_value = np.dot(self.problem.feature_rep, self.w)
        error = (est_value - self.state_values)
        error_squared = error * error
        return np.sqrt(np.sum(self.d_mu[: -1] * error_squared[: -1]))

    def compute_step_size(self):
        return self.alpha

    def choose_behavior_action(self):
        return self.problem.select_behavior_action(self.state)

    def choose_target_action(self):
        return self.problem.select_target_action(self.state)

    def learn(self, *args):
        raise NotImplementedError
