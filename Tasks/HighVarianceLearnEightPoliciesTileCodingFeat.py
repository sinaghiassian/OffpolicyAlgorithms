import numpy as np
from Tasks.LearnEightPoliciesTileCodingFeat import LearnEightPoliciesTileCodingFeat


class HighVarianceLearnEightPoliciesTileCodingFeat(LearnEightPoliciesTileCodingFeat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.RANDOM_PROB = 0.97

    def select_behavior_action(self, s):
        random_num = np.random.random()
        x, y = self.get_xy(s)
        if x == 1 and (y == 1 or y == 8):
            if random_num < self.RANDOM_PROB:
                return self.ACTION_LEFT
            else:
                return np.random.choice([self.ACTION_UP, self.ACTION_RIGHT, self.ACTION_DOWN])
        if x == 8 and (y == 1 or y == 8):
            if random_num < self.RANDOM_PROB:
                return self.ACTION_RIGHT
            else:
                return np.random.choice([self.ACTION_UP, self.ACTION_LEFT, self.ACTION_DOWN])
        return np.random.choice([self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT])

    def get_mu(self, s, a):
        x, y = self.get_xy(s)
        if x == 1 and (y == 1 or y == 8):
            if a == self.ACTION_LEFT:
                return self.RANDOM_PROB
            else:
                return (1 - self.RANDOM_PROB) / 3.0
        if x == 8 and (y == 1 or y == 8):
            if a == self.ACTION_RIGHT:
                return self.RANDOM_PROB
            else:
                return (1 - self.RANDOM_PROB) / 3.0

        return super().get_mu(s, a)
