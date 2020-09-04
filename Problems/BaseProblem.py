from abc import abstractmethod
import numpy as np


class BaseProblem:
    def __init__(self, **kwargs):
        self.run_number = kwargs.get('run_number', 0)
        self.num_steps = 5000
        self.feature_rep = None
        self.stacked_feature_rep = None  # If learning more than one target policy at the same time
        self.num_features = None
        self.GAMMA = None
        self.behavior_dist = None
        self.state_values = None
        self.num_policies = None

    @abstractmethod
    def get_active_policies(self, s):
        raise NotImplementedError

    def stack_feature_rep(self):
        stacked_feature_rep = np.zeros((self.num_policies, self.feature_rep.shape[1], self.feature_rep.shape[0]))
        for i in range(self.feature_rep.shape[0]):
            stacked_x = np.tile(self.feature_rep[i, :], [self.num_policies, 1])
            stacked_feature_rep[:, :, i] = stacked_x
        return stacked_feature_rep

    @abstractmethod
    def load_feature_rep(self):
        raise NotImplementedError

    @abstractmethod
    def get_state_feature_rep(self, s):
        raise NotImplementedError

    @abstractmethod
    def create_feature_rep(self):
        raise NotImplementedError

    @abstractmethod
    def select_target_action(self, s, policy_id=0):
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

    @abstractmethod
    def load_behavior_dist(self):
        return self.behavior_dist

    @abstractmethod
    def load_state_values(self):
        return self.state_values
