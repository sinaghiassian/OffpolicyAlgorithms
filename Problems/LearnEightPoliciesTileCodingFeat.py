from abc import abstractmethod, ABC

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


class Policy(ABC):
    @abstractmethod
    def get_action(self, s):
        raise NotImplementedError

    @abstractmethod
    def get_probability(self, s, a):
        raise NotImplementedError

    @abstractmethod
    def get_possible_action(self, s):
        raise NotImplementedError


class FourRoomGridWorldPolicy(Policy, FourRoomGridWorld):
    def __init__(self, policy_number):
        FourRoomGridWorld.__init__(self)
        self._policy_number = policy_number
        self._state_actions = ImmutableDict(
            {
                0: [
                    ['0 <= y <= 3 and 2 <= x <= 4', [self.ACTION_DOWN, self.ACTION_RIGHT]],
                    ['3 >= y >= 0 == x', [self.ACTION_UP, self.ACTION_RIGHT]],
                    ['0 <= y <= 4 and x == 1', [self.ACTION_RIGHT]],
                    ['x == 5 and y == 1', [self.ACTION_DOWN]],
                    ['4 == y and 2 <= x <= 4', [self.ACTION_DOWN]],
                    ['4 == y and x == 0', [self.ACTION_UP]]
                ]
            })

    def _eval(self, condition, x, y):
        return eval(condition.replace('x', str(x)).replace('y', str(y)))

    def get_probability(self, s, a):
        for condition, possible_actions in self._state_actions[self._policy_number].items():
            x, y = s
            if self._eval(condition, x, y) and a in possible_actions:
                return len(possible_actions) / 1
        return 0

    def get_possible_action(self, s):
        for condition, possible_actions in self._state_actions[self._policy_number].items():
            x, y = s
            if self._eval(condition, x, y):
                return possible_actions
        return None

    def get_action(self, s):
        for condition, possible_actions in self._state_actions[self._policy_number]:
            x, y = s
            if self._eval(condition, x, y):
                return random.choice(possible_actions)
        return None


if __name__ == "__main__":
    actions = {
        0: 'up',
        1: 'down',
        2: 'right',
        3: 'left',
    }
    env = FourRoomGridWorld()
    policy0 = FourRoomGridWorldPolicy(0)
    state = env.reset()
    env.render()
    is_terminal = False
    for step in range(40):
        a = policy0.get_action(state)
        next_state, r, is_terminal, info = env.step(a)
        x, y, is_rand, selected_action = info.values()
        print(
            f'sept:{step}, '
            f'state:({state[0]},{state[1]}), '
            f'action: {actions[a]}, '
            f'environment_action: {actions[selected_action]}, '
            f'next_state:({next_state[0]},{next_state[1]}), '
            f'stochasticity:{is_rand}, '
            f'terminal:{is_terminal}'
        )
        state = next_state
        env.render()
        if is_terminal:
            break
