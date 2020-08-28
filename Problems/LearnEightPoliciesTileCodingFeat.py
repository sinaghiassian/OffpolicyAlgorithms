import numpy as np

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
