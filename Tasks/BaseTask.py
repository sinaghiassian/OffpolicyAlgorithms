from abc import abstractmethod
import numpy as np


class BaseTask:
    def __init__(self, **kwargs):
        self.run_number = kwargs.get('run_number', 0)
        self.num_steps = None
        self.feature_rep = None
        self.stacked_feature_rep = None  # If learning more than one target policy at the same time
        self.num_features = None
        self.GAMMA = None
        self.behavior_dist = None
        self.state_values = None
        self.num_policies = None
        self.ABTD_xi_zero = None
        self.ABTD_xi_max = None

    def stack_feature_rep(self):
        stacked_feature_rep = np.zeros((self.num_policies, self.feature_rep.shape[1], self.feature_rep.shape[0]))
        for i in range(self.feature_rep.shape[0]):
            stacked_x = np.tile(self.feature_rep[i, :], [self.num_policies, 1])
            stacked_feature_rep[:, :, i] = stacked_x
        return stacked_feature_rep

    def get_active_policies(self, s):
        ...

    def get_terminal_policies(self, s):
        ...

    def generate_behavior_dist(self, total_steps):
        ...

    @staticmethod
    def num_of_policies():
        ...

    @abstractmethod
    def load_feature_rep(self):
        ...

    @abstractmethod
    def get_state_feature_rep(self, s):
        ...

    @abstractmethod
    def create_feature_rep(self):
        ...

    @abstractmethod
    def select_target_action(self, s, policy_id=0):
        ...

    @abstractmethod
    def select_behavior_action(self, s):
        ...

    @abstractmethod
    def get_pi(self, s, a):
        ...

    @abstractmethod
    def get_mu(self, s, a):
        ...

    @abstractmethod
    def load_behavior_dist(self):
        return self.behavior_dist

    @abstractmethod
    def load_state_values(self):
        return self.state_values

    def __str__(self):
        return f'task:{type(self).__name__}'
