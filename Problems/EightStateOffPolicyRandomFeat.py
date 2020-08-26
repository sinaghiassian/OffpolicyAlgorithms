import numpy as np

from Environments.Chain import Chain
from Problems.BaseProblem import BaseProblem


class EightStateOffPolicyRandomFeat(BaseProblem, Chain):
    def __init__(self, run_number, n=8):
        BaseProblem.__init__(self, run_number)
        Chain.__init__(self)
        self.N = n
        self.num_features = self.feature_rep.shape[1]
        self.num_steps = 100
        self.GAMMA = 0.9
        self.behavior_dist = np.zeros(self.N + 1)
        self.state_values = np.zeros(self.N + 1)

    def create_feature_rep(self):
        num_ones = 3
        num_zeros = self.num_features - num_ones
        for i in range(self.N):
            random_arr = (np.array([0] * num_zeros + [1] * num_ones))
            np.random.shuffle(random_arr)
            self.feature_rep[i, :] = random_arr

    def get_state_feature_rep(self, state):
        return self.feature_rep[state, :]

    @property
    def get_behavior_dist(self):
        self.behavior_dist = np.load('Resource/d_mu.npy')
        return self.behavior_dist

    @property
    def get_state_value(self):
        self.state_values = np.load('Resource/state_values.npy')
        return self.state_values

    def select_behavior_action(self, s):
        if s < self.N / 2:
            return self.RIGHT_ACTION
        else:
            return np.random.choice([self.RIGHT_ACTION, self.RETREAT_ACTION])

    def select_target_action(self, s):
        return self.RIGHT_ACTION

    def get_pi(self, s, a):
        if a == self.RIGHT_ACTION:
            return 1
        else:
            return 0

    def get_mu(self, s, a):
        if s < self.N / 2:
            if a == self.RIGHT_ACTION:
                return 1
            else:
                return 0
        elif s >= self.N / 2:
            return 0.5
        else:
            raise AssertionError
