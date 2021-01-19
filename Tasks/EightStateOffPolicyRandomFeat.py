import numpy as np

from Environments.Chain import Chain
from Tasks.BaseTask import BaseTask


class EightStateOffPolicyRandomFeat(BaseTask, Chain):
    def __init__(self, **kwargs):
        BaseTask.__init__(self, **kwargs)
        Chain.__init__(self)
        self.N = kwargs.get('n', 8)
        self.feature_rep = self.load_feature_rep()
        self.num_features = self.feature_rep.shape[1]
        self.num_steps = 10000
        self.GAMMA = 0.9
        self.behavior_dist = self.load_behavior_dist()
        self.state_values = self.load_state_values()
        self.num_policies = 1
        self.ABTD_si_zero = 1
        self.ABTD_si_max = 2

    def load_feature_rep(self):
        return np.load(f'Resources/{self.__class__.__name__}/feature_rep.npy')[:, :, self.run_number]

    def create_feature_rep(self):
        num_ones = 3
        num_zeros = self.num_features - num_ones
        for i in range(self.N):
            random_arr = (np.array([0] * num_zeros + [1] * num_ones))
            np.random.shuffle(random_arr)
            self.feature_rep[i, :] = random_arr

    def get_state_feature_rep(self, s):
        return self.feature_rep[s, :]

    def load_behavior_dist(self):
        self.behavior_dist = np.load(f'Resources/{self.__class__.__name__}/d_mu.npy')
        return self.behavior_dist

    def load_state_values(self):
        self.state_values = np.load(f'Resources/{self.__class__.__name__}/state_values.npy')
        return self.state_values

    def select_behavior_action(self, s):
        if s < self.N / 2:
            return self.RIGHT_ACTION
        else:
            return np.random.choice([self.RIGHT_ACTION, self.RETREAT_ACTION])

    def select_target_action(self, s, policy_id=0):
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
