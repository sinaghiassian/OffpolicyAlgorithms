import numpy as np
# import gym
# from gym import utils
# import sys
# from io import StringIO


class Chain:
    def __init__(self, states_number: int = 8, start_state_number: int = 4, **kwargs):
        assert start_state_number < states_number, "start states numbers should be less than state number"

        self._states_number = states_number
        self._start_state_number = start_state_number
        self._terminal = self._states_number
        self._state = None
        self.RIGHT_ACTION = 0
        self.RETREAT_ACTION = 1

    def reset(self):
        self._state = np.random.randint(0, self._start_state_number)
        return self._state

    def step(self, action):
        if action == self.RETREAT_ACTION:
            return self._terminal, 0, True, {}

        next_state = self._state + 1
        if next_state == self._terminal:
            return self._terminal, 1, True, {}

        self._state = next_state
        return self._state, 0, False, {}

    # def render(self, mode='human'):
    #     if mode == 'human':
    #         outfile = StringIO() if mode == 'ansi' else sys.stdout
    #         corridor_map = [
    #             str(i) if i > self._start_state_number
    #             else utils.colorize(str(i), "blue", highlight=False)
    #             for i in range(self._states_number)
    #         ]
    #         corridor_map.append(utils.colorize("T", "red", highlight=False))
    #         corridor_map[self._state] = utils.colorize(corridor_map[self._state], "green", highlight=True)
    #
    #         outfile.write(f'{"|".join(corridor_map)}\n')
