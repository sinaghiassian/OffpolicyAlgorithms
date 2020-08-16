import numpy as np


class FiveStateChainProb:
    def __init__(self, n=5):
        self.N = n
        self.feature_rep = {'tabular': np.zeros((self.N + 1, self.N)),
                            'dependent': np.zeros((self.N + 1, self.N)),
                            'inverted': np.zeros((self.N + 1, self.N))}
        self.num_steps = 3000
        self.GAMMA = 1.0
        self.behavior_dist = np.zeros(n + 1)
        self.state_values = np.zeros(n + 1)

    def get_num_steps(self):
        return self.num_steps

    def get_gamma(self):
        return self.GAMMA

    def get_behavior_dist(self):
        self.behavior_dist = np.array([.11111, 0.22222, 0.33333, 0.22222, 0.11111, 0.0])
        return self.behavior_dist

    def get_state_values(self):
        self.state_values = np.array([1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0, 4.0 / 6.0, 5.0 / 6.0, 0.0])
        return self.state_values

    def get_feature_rep(self, rep_type):
        tabular_features = np.zeros((self.N + 1, self.N))
        inverted_features = np.zeros((self.N + 1, self.N))
        n_feats_dependent = int(np.floor(self.N / 2) + 1)
        dependent_features = np.zeros((self.N + 1, n_feats_dependent))

        m = np.eye(self.N)
        tabular_features[: self.N] = m

        m = np.ones((self.N, self.N)) - np.eye(self.N)
        inverted_features[: self.N] = (m.T / np.linalg.norm(m, axis=1)).T

        idx = 0
        for i in range(n_feats_dependent):
            dependent_features[idx, 0: i + 1] = 1
            idx = idx + 1
        for i in range(n_feats_dependent - 1, 0, -1):
            dependent_features[idx, -i:] = 1
            idx = idx + 1
        dependent_features[: self.N] = (
                dependent_features[:self.N].T / np.linalg.norm(dependent_features[:self.N], axis=1)).T

        self.feature_rep['tabular'] = tabular_features
        self.feature_rep['inverted'] = inverted_features
        self.feature_rep['dependent'] = dependent_features
        return self.feature_rep[rep_type]
