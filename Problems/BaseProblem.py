from abc import abstractmethod, ABC
import numpy as np


class BaseProblem:
    def __init__(self, run_number):
        self.run_number = run_number
        self.num_steps = 5000
        self.feature_rep = np.load('Resource/feature_rep.npy')[:, :, run_number]
        self.num_features = self.feature_rep.shape[1]
        self.GAMMA = None
        self.behavior_dist = None
        self.state_values = None

    @abstractmethod
    def get_state_feature_rep(self, state):
        raise NotImplementedError

    @abstractmethod
    def create_feature_rep(self):
        raise NotImplementedError

    @abstractmethod
    def select_target_action(self, s):
        raise NotImplementedError

    @abstractmethod
    def select_behavior_action(self, s):
        raise NotImplementedError

    @abstractmethod
    def get_pi(self, s, a):
        raise NotImplementedError

    @abstractmethod
    def get_mu(self, s, a):
        raise NotImplementedError

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
