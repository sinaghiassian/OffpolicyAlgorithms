import numpy as np
import random

from Environments.FourRoomGridWorld import FourRoomGridWorld
from Tasks.BaseTask import BaseTask
from utils import ImmutableDict


class LearnEightPoliciesTileCodingFeat(BaseTask, FourRoomGridWorld):
    def __init__(self, **kwargs):
        BaseTask.__init__(self)
        FourRoomGridWorld.__init__(self)
        self.feature_rep = self.load_feature_rep()
        self.num_features = self.feature_rep.shape[1]
        self.num_steps = kwargs.get('num_steps', 50000)
        self.GAMMA = 0.9
        self.behavior_dist = self.load_behavior_dist()
        self.state_values = self.load_state_values()
        self.ABTD_si_zero = 1
        self.ABTD_si_max = 4

        self.optimal_policies = ImmutableDict(
            {
                0: [
                    [lambda x, y: 0 <= x <= 3 and 2 <= y <= 4, [self.ACTION_DOWN, self.ACTION_RIGHT]],
                    [lambda x, y: 3 >= x >= 0 == y, [self.ACTION_UP, self.ACTION_RIGHT]],
                    [lambda x, y: 0 <= x <= 4 and y == 1, [self.ACTION_RIGHT]],
                    [lambda x, y: x == self.hallways[1][0] and y == self.hallways[1][1], [self.ACTION_DOWN]],
                    [lambda x, y: 4 == x and 2 <= y <= 4, [self.ACTION_DOWN]],
                    [lambda x, y: 4 == x and y == 0, [self.ACTION_UP]]
                ],
                1: [
                    [lambda x, y: 2 <= x <= 4 and 0 <= y <= 3, [self.ACTION_LEFT, self.ACTION_UP]],
                    [lambda x, y: x == 0 and 0 <= y <= 3, [self.ACTION_RIGHT, self.ACTION_UP]],
                    [lambda x, y: x == 1 and 0 <= y <= 4, [self.ACTION_UP]],
                    [lambda x, y: x == self.hallways[0][0] and y == self.hallways[0][1], [self.ACTION_LEFT]],
                    [lambda x, y: 2 <= x <= 4 and y == 4, [self.ACTION_LEFT]],
                    [lambda x, y: x == 0 and y == 4, [self.ACTION_RIGHT]],
                ],
                2: [
                    [lambda x, y: 2 <= x <= 4 and 7 <= y <= 10, [self.ACTION_LEFT, self.ACTION_DOWN]],
                    [lambda x, y: x == 0 and 7 <= y <= 10, [self.ACTION_RIGHT, self.ACTION_DOWN]],
                    [lambda x, y: x == 1 and 6 <= y <= 10, [self.ACTION_DOWN]],
                    [lambda x, y: x == self.hallways[2][0] and y == self.hallways[2][1], [self.ACTION_LEFT]],
                    [lambda x, y: 2 <= x <= 4 and y == 6, [self.ACTION_LEFT]],
                    [lambda x, y: x == 0 and y == 6, [self.ACTION_RIGHT]],
                ],
                3: [
                    [lambda x, y: 0 <= x <= 3 and 6 <= y <= 7, [self.ACTION_UP, self.ACTION_RIGHT]],
                    [lambda x, y: 0 <= x <= 3 and 9 <= y <= 10, [self.ACTION_DOWN, self.ACTION_RIGHT]],
                    [lambda x, y: 0 <= x <= 4 and y == 8, [self.ACTION_RIGHT]],
                    [lambda x, y: x == self.hallways[1][0] and y == self.hallways[1][1], [self.ACTION_UP]],
                    [lambda x, y: x == 4 and 6 <= y <= 7, [self.ACTION_UP]],
                    [lambda x, y: x == 4 and 9 <= y <= 10, [self.ACTION_DOWN]]
                ],
                4: [
                    [lambda x, y: 10 >= x >= 7 >= y >= 5, [self.ACTION_LEFT, self.ACTION_UP]],
                    [lambda x, y: 7 <= x <= 10 and 9 <= y <= 10, [self.ACTION_LEFT, self.ACTION_DOWN]],
                    [lambda x, y: 6 <= x <= 10 and y == 8, [self.ACTION_LEFT]],
                    [lambda x, y: x == self.hallways[3][0] and y == self.hallways[3][1], [self.ACTION_UP]],
                    [lambda x, y: x == 6 and 5 <= y <= 7, [self.ACTION_UP]],
                    [lambda x, y: x == 6 and 9 <= y <= 10, [self.ACTION_DOWN]]
                ],
                5: [
                    [lambda x, y: 6 <= x <= 7 and 6 <= y <= 10, [self.ACTION_RIGHT, self.ACTION_DOWN]],
                    [lambda x, y: 9 <= x <= 10 and 6 <= y <= 10, [self.ACTION_DOWN, self.ACTION_LEFT]],
                    [lambda x, y: x == 8 and 5 <= y <= 10, [self.ACTION_DOWN]],
                    [lambda x, y: x == self.hallways[2][0] and y == self.hallways[2][1], [self.ACTION_RIGHT]],
                    [lambda x, y: 6 <= x <= 7 and y == 5, [self.ACTION_RIGHT]],
                    [lambda x, y: 9 <= x <= 10 and y == 5, [self.ACTION_LEFT]]
                ],
                6: [
                    [lambda x, y: 6 <= x <= 7 and 0 <= y <= 2, [self.ACTION_UP, self.ACTION_RIGHT]],
                    [lambda x, y: 9 <= x <= 10 and 0 <= y <= 2, [self.ACTION_UP, self.ACTION_LEFT]],
                    [lambda x, y: x == 8 and 0 <= y <= 3, [self.ACTION_UP]],
                    [lambda x, y: x == self.hallways[0][0] and y == self.hallways[0][1], [self.ACTION_RIGHT]],
                    [lambda x, y: 6 <= x <= 7 and y == 3, [self.ACTION_RIGHT]],
                    [lambda x, y: 9 <= x <= 10 and y == 3, [self.ACTION_LEFT]]
                ],
                7: [
                    [lambda x, y: 7 <= x <= 10 and 2 <= y <= 3, [self.ACTION_DOWN, self.ACTION_LEFT]],
                    [lambda x, y: 7 <= x <= 10 and y == 0, [self.ACTION_UP, self.ACTION_LEFT]],
                    [lambda x, y: 6 <= x <= 10 and y == 1, [self.ACTION_LEFT]],
                    [lambda x, y: x == self.hallways[3][0] and y == self.hallways[3][1], [self.ACTION_DOWN]],
                    [lambda x, y: x == 6 and 2 <= y <= 3, [self.ACTION_DOWN]],
                    [lambda x, y: x == 6 and y == 0, [self.ACTION_UP]]
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
        self.policy_terminal_condition = ImmutableDict(
            {
                0: lambda x, y: x == self.hallways[0][0] and y == self.hallways[0][1],
                1: lambda x, y: x == self.hallways[1][0] and y == self.hallways[1][1],
                2: lambda x, y: x == self.hallways[1][0] and y == self.hallways[1][1],
                3: lambda x, y: x == self.hallways[2][0] and y == self.hallways[2][1],
                4: lambda x, y: x == self.hallways[2][0] and y == self.hallways[2][1],
                5: lambda x, y: x == self.hallways[3][0] and y == self.hallways[3][1],
                6: lambda x, y: x == self.hallways[3][0] and y == self.hallways[3][1],
                7: lambda x, y: x == self.hallways[0][0] and y == self.hallways[0][1]
            }
        )
        self.num_policies = LearnEightPoliciesTileCodingFeat.num_of_policies()
        self.stacked_feature_rep = self.stack_feature_rep()
        self._active_policies_cache = {}

    @staticmethod
    def num_of_policies():
        return 8

    def get_terminal_policies(self, s):
        x, y = self.get_xy(s)
        terminal_policies = np.zeros(self.num_policies)
        for policy_id, condition in self.policy_terminal_condition.items():
            if condition(x, y):
                terminal_policies[policy_id] = 1
        return terminal_policies

    def get_state_index(self, x, y):
        return int(y * np.sqrt(self.feature_rep.shape[0]) + x)

    def get_probability(self, policy_number, s, a):
        x, y = self.get_xy(s)
        probability = 0.0
        for condition, possible_actions in self.optimal_policies[policy_number]:
            if condition(x, y):
                if a in possible_actions:
                    probability = 1.0 / len(possible_actions)
                    break
        return probability

    def select_target_action(self, s, policy_id=0):
        x, y = self.get_xy(s)
        a = self.default_actions[policy_id]
        for condition, possible_actions in self.optimal_policies[policy_id]:
            if condition(x, y):
                a = random.choice(possible_actions)
                break
        return a

    def get_active_policies(self, s):
        if s in self._active_policies_cache:
            return self._active_policies_cache[s]
        x, y = self.get_xy(s)
        active_policy_vec = np.zeros(self.num_policies, dtype=int)
        for policy_number, policy_values in self.optimal_policies.items():
            for (condition, _) in policy_values:
                if condition(x, y):
                    active_policy_vec[policy_number] = 1
                    break
        self._active_policies_cache[s] = active_policy_vec
        return active_policy_vec

    def load_feature_rep(self):
        return np.load(f'Resources/{self.__class__.__name__}/feature_rep.npy')

    def get_state_feature_rep(self, s):
        return self.feature_rep[s, :]

    def create_feature_rep(self):
        ...

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
        for policy_id, i in enumerate(self.get_active_policies(s)):
            if i:
                pi_vec[policy_id] = self.get_probability(policy_id, s, a)
        return pi_vec
