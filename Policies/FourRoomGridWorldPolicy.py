import random
from Environments.FourRoomGridWorld import FourRoomGridWorld
from Policies.BasePolicy import BasePolicy
from utils import ImmutableDict


class FourRoomGridWorldPolicy(BasePolicy, FourRoomGridWorld):
    def __init__(self):
        FourRoomGridWorld.__init__(self)
        self._state_actions = ImmutableDict(
            {
                0: [
                    ['0 <= y <= 3 and 2 <= x <= 4', [self.ACTION_DOWN, self.ACTION_RIGHT]],
                    ['3 >= y >= 0 == x', [self.ACTION_UP, self.ACTION_RIGHT]],
                    ['0 <= y <= 4 and x == 1', [self.ACTION_RIGHT]],
                    ['x == 5 and y == 1', [self.ACTION_DOWN]],
                    ['4 == y and 2 <= x <= 4', [self.ACTION_DOWN]],
                    ['4 == y and x == 0', [self.ACTION_UP]]
                ],
                1: [
                    ['2 <= y <= 4 and 0 <= x <= 3', [self.ACTION_LEFT, self.ACTION_UP]],
                    ['y == 0 and 0 <= x <= 3', [self.ACTION_RIGHT, self.ACTION_UP]],
                    ['y == 1 and 0 <= x <= 4', [self.ACTION_UP]],
                    ['x == 1 and y == 5', [self.ACTION_LEFT]],
                    ['2 <= y <= 4 and x == 4', [self.ACTION_LEFT]],
                    ['y == 0 and x == 4', [self.ACTION_RIGHT]],
                ]
            })

    @staticmethod
    def _eval(condition, x, y):
        return eval(condition.replace('x', str(x)).replace('y', str(y)))

    def get_probability(self, policy_number, s, a):
        for condition, possible_actions in self._state_actions[policy_number].items():
            x, y = s
            if self._eval(condition, x, y) and a in possible_actions:
                return len(possible_actions) / 1
        return 0

    def get_possible_action(self, policy_number, s):
        for condition, possible_actions in self._state_actions[policy_number].items():
            x, y = s
            if self._eval(condition, x, y):
                return possible_actions
        return None

    def get_action(self, policy_number, s):
        for condition, possible_actions in self._state_actions[policy_number]:
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
    policy = FourRoomGridWorldPolicy()
    state = env.reset()
    env.render()
    is_terminal = False
    for step in range(40):
        a = policy.get_action(1, state)
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
