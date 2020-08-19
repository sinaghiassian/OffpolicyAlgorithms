from abc import abstractmethod
import numpy as np


class BaseProblem:
    def __init__(self):
        self.num_steps = 5000
        self.num_features = None
        self.GAMMA = None
        self.behavior_dist = None
        self.state_values = None

    @abstractmethod
    def create_feature_rep(self):
        return NotImplementedError

    @abstractmethod
    def select_target_action(self, s):
        return NotImplementedError

    @abstractmethod
    def select_behavior_action(self, s):
        return NotImplementedError

    @property
    def get_num_steps(self):
        return self.num_steps

    @property
    def get_gamma(self):
        return self.GAMMA

    @property
    def get_behavior_dist(self):
        return self.behavior_dist

    @property
    def get_state_value(self):
        return self.state_values
