import numpy as np


class EightStateOffPolicyRandomFeat:
    def __init__(self, n=8):
        self.N = n
        self.num_features = 6
        self.feature_rep = np.zeros((self.N + 1, self.num_features))
        self.num_steps = 5000
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

    @property
    def get_num_steps(self):
        return self.num_steps

    @property
    def get_gamma(self):
        return self.GAMMA

    @property
    def get_behavior_dist(self):  # TODO: This is just a placeholder for now.
        self.behavior_dist = np.array([0, 0, 0])
        return self.behavior_dist

    @property
    def get_state_value(self):  # TODO: This is just a placeholder for now.
        self.state_values = np.array([0, 0, 0])
        return self.behavior_dist

    @property
    def get_feat_rep(self, run):
        return np.load('feature_rep.npy')[:, :, run]  # Todo: Clean this up.
        # self.create_feature_rep()
        # return self.feature_rep

