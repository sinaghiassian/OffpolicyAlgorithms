import numpy as np
import random

from Environments.FourRoomGridWorld import FourRoomGridWorld
from Problems.BaseProblem import BaseProblem


class LearnEightPoliciesTileCodingFeat(BaseProblem, FourRoomGridWorld):
    def __init__(self):
        BaseProblem.__init__(self)
        FourRoomGridWorld.__init__(self)
        self.feature_rep = self.load_feature_rep()
        self.num_features = self.feature_rep.shape[1]
        self.num_steps = 5000
        self.GAMMA = 0.9
        self.behavior_dist = self.load_behavior_dist()
        self.state_values = self.load_state_values()

    def load_feature_rep(self):
        return np.load(f'Resources/{self.__class__.__name__}/feature_rep.npy')[:, :]

    def load_behavior_dist(self):
        return np.load(f'Resources/{self.__class__.__name__}/d_mu.npy')

    def load_state_values(self):
        return np.load(f'Resources/{self.__class__.__name__}/state_values.npy')

    def get_state_feature_rep(self, s):
        return self.feature_rep[s, :]

    def select_behavior_action(self, s):
        return np.random.randint(0, self.num_actions)

    def choose_pi0_action(self, s):  # TODO: Check Sample.py in previous code version.
        x, y = s
        if 0 <= x <= 3 and 2 <= y <= 4:
            return random.choice([self.ACTION_DOWN, self.ACTION_RIGHT]), 0.5
        elif 3 >= x >= 0 == y:
            return random.choice([self.ACTION_UP, self.ACTION_RIGHT]), 0.5
        elif 0 <= x <= 4 and y == 1:
            return self.ACTION_RIGHT, 1.0
        elif x == 4 and y == 0:
            return self.ACTION_UP, 1.0
        elif s == self.hallways[1]:
            return self.ACTION_DOWN, 1.0

    def get_pi0_probability(self, s, a):
        x, y = s
        if 0 <= x <= 3 and 2 <= y <= 4 and a in [self.ACTION_DOWN, self.ACTION_RIGHT]:
            return 0.5
        elif s == self.hallways[1] and a == self.ACTION_DOWN:
            return 1.0
