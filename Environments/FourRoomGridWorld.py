import sys
from io import StringIO
import gym
import numpy as np
from gym import utils

BLOCK_NORMAL, BLOCK_WALL, BLOCK_HALLWAY = 0, 1, 2

four_room_map = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 1, 2, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]


class FourRoomGridWorld(gym.Env):
    def __init__(self, stochasticity_fraction=0.0):
        self._grid = np.transpose(np.flip(np.array(four_room_map, dtype=np.uint8), axis=0)[1:-1, 1:-1])
        self._max_row, self._max_col = self._grid.shape
        self._normal_tiles = np.where(self._grid == BLOCK_NORMAL)
        self._state = None
        self._color = {
            BLOCK_NORMAL: lambda c: utils.colorize(c, "white", highlight=True),
            BLOCK_WALL: lambda c: utils.colorize(c, "gray", highlight=True),
            BLOCK_HALLWAY: lambda c: utils.colorize(c, "green", highlight=True),
        }
        self.ACTION_UP, self.ACTION_DOWN, self.ACTION_RIGHT, self.ACTION_LEFT = 0, 1, 2, 3
        self.num_actions = 4
        self._stochasticity_fraction = stochasticity_fraction
        self.hallways = {
            0: (5, 1),
            1: (1, 5),
            2: (5, 8),
            3: (8, 4)
        }

    def reset(self):
        # if random_agent_start:
        #     rnd = np.random.choice(len(self._normal_tiles[0]))
        #     self._state = (self._normal_tiles[0][rnd], self._normal_tiles[1][rnd])
        # else:
        self._state = (0, 0)
        return self.get_state_index(*self._state)

    def step(self, action):
        x, y = self._state
        is_stochastic_selected = False
        if self._stochasticity_fraction >= np.random.uniform():
            action_probability = [1 / (self.num_actions - 1) if i != action else 0 for i in range(self.num_actions)]
            action = np.random.choice(self.num_actions, 1, p=action_probability)[0]
            is_stochastic_selected = True
        x_p, y_p = self._next(action, *self._state)
        is_done = self._grid[x_p, y_p] == BLOCK_HALLWAY
        reward = 1 if is_done else 0
        self._state = (x_p, y_p)
        return self.get_state_index(*self._state), reward, False, {
            'x': x, 'y': y,
            'x_p': x_p, 'y_p': y_p,
            'is_stochastic_selected': is_stochastic_selected,
            'selected_action': action}

    def render(self, mode='human', show_state_numbers=False):
        if mode == 'human':
            outfile = sys.stdout
            img = [
                [self._color[b]('  ')
                 for x, b
                 in enumerate(line)]
                for y, line in enumerate(four_room_map)]
            img[self._max_row - self._state[1]][self._state[0] + 1] = utils.colorize('  ', "red",
                                                                                     highlight=True)
            for line in img:
                outfile.write(f'{"".join(line)}\n')
            outfile.write('\n')

    def get_xy(self, state):
        return (state % self._max_row),(state // self._max_col)

    def get_state_index(self, x, y):
        return y * self._max_col + x

    def _next(self, action, x, y):

        def move(current_x, current_y, next_x, next_y):
            if next_y < 0 or next_x < 0:
                return current_x, current_y
            if next_y >= self._max_col or next_x >= self._max_row:
                return current_x, current_y
            if self._grid[next_x, next_y] == BLOCK_WALL:
                return current_x, current_y
            return next_x, next_y

        switcher = {
            self.ACTION_DOWN: lambda x, y: move(x, y, x, y - 1),
            self.ACTION_RIGHT: lambda x, y: move(x, y, x + 1, y),
            self.ACTION_UP: lambda x, y: move(x, y, x, y + 1),
            self.ACTION_LEFT: lambda x, y: move(x, y, x - 1, y),
        }
        move_func = switcher.get(action)
        return move_func(x, y)
