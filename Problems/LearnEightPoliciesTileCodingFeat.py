import numpy as np
import random

from Environments.FourRoomGridWorld import FourRoomGridWorld
from Problems.BaseProblem import BaseProblem
from utils import ImmutableDict


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

        self.optimal_policies = ImmutableDict(
            {
                0: [
                    ['0 <= x <= 3 and 2 <= y <= 4', [self.ACTION_DOWN, self.ACTION_RIGHT]],
                    ['3 >= x >= 0 == y', [self.ACTION_UP, self.ACTION_RIGHT]],
                    ['0 <= x <= 4 and y == 1', [self.ACTION_RIGHT]],
                    ['x == hall[1][0] and y == hall[1][1]', [self.ACTION_DOWN]],
                    ['4 == x and 2 <= y <= 4', [self.ACTION_DOWN]],
                    ['4 == x and y == 0', [self.ACTION_UP]]
                ],
                1: [
                    ['2 <= x <= 4 and 0 <= y <= 3', [self.ACTION_LEFT, self.ACTION_UP]],
                    ['x == 0 and 0 <= y <= 3', [self.ACTION_RIGHT, self.ACTION_UP]],
                    ['x == 1 and 0 <= y <= 4', [self.ACTION_UP]],
                    ['y == hall[0][0] and x == hall[0][1]', [self.ACTION_LEFT]],
                    ['2 <= x <= 4 and y == 4', [self.ACTION_LEFT]],
                    ['x == 0 and y == 4', [self.ACTION_RIGHT]],
                ],
                2: [
                    ['2 <= x <= 4 and 7 <= y <= 10', [self.ACTION_LEFT, self.ACTION_DOWN]],
                    ['x == 0 and 7 <= y <= 10', [self.ACTION_RIGHT, self.ACTION_DOWN]],
                    ['x == 1 and 6 <= y <= 10', [self.ACTION_DOWN]],
                    ['x == hall[2][0] and y == hall[2][1]', [self.ACTION_LEFT]],
                    ['2 <= x <= 4 and y == 6', [self.ACTION_LEFT]],
                    ['x == 0 and y == 6', [self.ACTION_RIGHT]],
                ],
                3: [
                    ['0 <= x <= 3 and 6 <= y <= 7', [self.ACTION_UP, self.ACTION_RIGHT]],
                    ['0 <= x <= 3 and 9 <= y <= 10', [self.ACTION_DOWN, self.ACTION_RIGHT]],
                    ['0 <= x <= 4 and y == 8', [self.ACTION_RIGHT]],
                    ['x == hall[1][0] and y == hall[1][1]', [self.ACTION_UP]],
                    ['x == 4 and 6 <= y <= 7', [self.ACTION_UP]],
                    ['x == 4 and 9 <= y <= 10', [self.ACTION_DOWN]]
                ],
                4: [
                    ['10 >= x >= 7 >= y and y >= 5', [self.ACTION_LEFT, self.ACTION_UP]],
                    ['7 <= x <= 10 and 9 <= y <= 10', [self.ACTION_LEFT, self.ACTION_DOWN]],
                    ['6 <= x <= 10 and y == 8', [self.ACTION_LEFT]],
                    ['x == hall[3][0] and y == hall[3][1]', [self.ACTION_UP]],
                    ['x == 6 and 5 <= y <= 7', [self.ACTION_UP]],
                    ['x == 6 and 9 <= y <= 10', [self.ACTION_DOWN]]
                ],
                5: [
                    ['6 <= x <= 7 and 6 <= y <= 1', [self.ACTION_RIGHT, self.ACTION_DOWN]],
                    ['9 <= x <= 10 and 6 <= y <= 10', [self.ACTION_DOWN, self.ACTION_LEFT]],
                    ['x == 8 and 5 <= y <= 10', [self.ACTION_DOWN]],
                    ['x == hall[2][0] and y == hall[2][1]', [self.ACTION_RIGHT]],
                    ['6 <= x <= 7 and y == 5', [self.ACTION_RIGHT]],
                    ['9 <= x <= 10 and y == 5', [self.ACTION_LEFT]]
                ],
                6: [
                    ['6 <= x <= 7 and 0 <= y <= 2', [self.ACTION_UP, self.ACTION_RIGHT]],
                    ['9 <= x <= 10 and 0 <= y <= 2', [self.ACTION_UP, self.ACTION_LEFT]],
                    ['x == 8 and 0 <= y <= 3', [self.ACTION_UP]],
                    ['x == hall[0][0] and y == hall[0][1]', [self.ACTION_RIGHT]],
                    ['6 <= x <= 7 and y == 3', [self.ACTION_RIGHT]],
                    ['9 <= x <= 10 and y == 3', [self.ACTION_LEFT]]
                ],
                7: [
                    ['7 <= x <= 10 and 2 <= y <= 3', [self.ACTION_DOWN, self.ACTION_LEFT]],
                    ['7 <= x <= 10 and y == 0', [self.ACTION_UP, self.ACTION_LEFT]],
                    ['6 <= x <= 10 and y == 1', [self.ACTION_LEFT]],
                    ['x == hall[3][0] and y == hall[3][1]', [self.ACTION_DOWN]],
                    ['x == 6 and 2 <= y <= 3', [self.ACTION_DOWN]],
                    ['x == 6 and y == 0', [self.ACTION_UP]]
                ]
            }
        )
        self.default_actions = ImmutableDict(
            {
                0: self.ACTION_RIGHT,
                1: self.ACTION_UP,
                2: self.ACTION_DOWN,
                3: self.ACTION_RIGHT,
                4: self.ACTION_LEFT,
                5: self.ACTION_DOWN,
                6: self.ACTION_UP,
                7: self.ACTION_LEFT
            }
        )
        self.num_policies = len(self.optimal_policies)

    def _eval(self, condition, x, y):
        return eval(condition, {'x': x, 'y': y, 'hall': self.hallways})

    def get_probability(self, policy_number, s, a):
        x, y = s
        probability = 0.0
        for condition, possible_actions in self.optimal_policies[policy_number]:
            if self._eval(condition, x, y):
                if a in possible_actions:
                    probability = 1.0 / len(possible_actions)
        return probability

    def select_target_action(self, s, policy_id=0):
        x, y = s
        a = self.default_actions[policy_id]
        for condition, possible_actions in self.optimal_policies[policy_id]:
            if self._eval(condition, x, y):
                a = random.choice(possible_actions)
        return a

    def get_active_policies(self, s):
        x, y = s
        active_policies = []
        for policy_number, (condition, _) in self.optimal_policies.items():
            if self._eval(condition, x, y):
                active_policies.append(policy_number)
        return active_policies

    def load_feature_rep(self):
        return np.load(f'Resources/{self.__class__.__name__}/feature_rep.npy')[:, :]

    def get_state_feature_rep(self, s):
        return self.feature_rep[s, :]

    def create_feature_rep(self):
        raise NotImplementedError

    def load_behavior_dist(self):
        return np.load(f'Resources/{self.__class__.__name__}/d_mu.npy')

    def load_state_values(self):
        return np.load(f'Resources/{self.__class__.__name__}/state_values.npy')

    def select_behavior_action(self, s):
        return np.random.randint(0, self.num_actions)

    def get_mu(self, s, a):
        return np.ones(self.num_policies) * (1.0 / self.num_actions)

    def get_pi(self, s, a):
        pi_vec = np.zeros(self.num_policies)
        for i in self.get_active_policies(s):
            pi_vec[i] = self.get_probability(i, s, a)
        return pi_vec
