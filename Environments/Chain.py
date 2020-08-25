import sys
from io import StringIO
import gym
import numpy as np
from enum import Enum
from gym import utils


class Chain(gym.Env):
    def __init__(self, states_number: int = 8, start_state_number: int = 4, **kwargs):
        assert start_state_number < states_number, "start states numbers should be less than state number"

        self._states_number = states_number
        self._start_state_number = start_state_number
        self._state = self.reset()
        self._terminal = self._states_number + 1
        self.RIGHT_ACTION = 0
        self.RETREAT_ACTION = 1

    def reset(self):
        return np.random.randint(0, self._start_state_number)

    def step(self, action):
        if action == self.RETREAT_ACTION:
            return self.reset(), 1, False, {}

        next_state = self._state + 1
        if next_state == self._terminal:
            return next_state, 1, True, {}

        self._state = next_state
        return self._state, 0, False, {}

    def render(self, mode='human'):
        if mode == 'human':
            outfile = StringIO() if mode == 'ansi' else sys.stdout
            corridor_map = [
                str(i) if i > self._start_state_number
                else utils.colorize(str(i), "blue", highlight=False)
                for i in range(self._states_number)
            ]
            corridor_map.append(utils.colorize("T", "red", highlight=False))
            corridor_map[self._state] = utils.colorize(corridor_map[self._state], "green", highlight=True)

            outfile.write(f'{"|".join(corridor_map)}\n')
