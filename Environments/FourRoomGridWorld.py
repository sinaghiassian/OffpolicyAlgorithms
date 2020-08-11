import sys
from io import StringIO
import gym
import numpy as np
from enum import Enum
from gym import utils


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class BlockType(Enum):
    Normal = 0
    Wall = 1
    Hallway = 2


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
    def __init__(self):
        self._grid = np.array(four_room_map, dtype=np.uint8)
        self._max_row, self._max_col = self._grid.shape
        self._normal_tiles = np.where(self._grid == BlockType.Normal.value)
        self._state = self.reset()
        self._fill_char = '  '
        self._color = {
            BlockType.Normal.value: utils.colorize(self._fill_char, "white", highlight=True),
            BlockType.Wall.value: utils.colorize(self._fill_char, "gray", highlight=True),
            BlockType.Hallway.value: utils.colorize(self._fill_char, "green", highlight=True),
        }

    def reset(self, random_agent_start=True, **kwargs):
        if random_agent_start:
            rnd = np.random.choice(len(self._normal_tiles[0]))
            self._state = (self._normal_tiles[0][rnd], self._normal_tiles[1][rnd])
        else:
            assert 'x' in kwargs, 'x is required'
            assert 'y' in kwargs, 'y is required'
            self._state = (kwargs['x'], kwargs['y'])
        return self._state

    def step(self, action: Action):
        next_x, next_y = self._next(action, *self._state)
        is_done = self._grid[next_x, next_y] == BlockType.Hallway.value
        reward = -1
        self._state = (next_x, next_y)
        return self._state, reward, is_done, {}

    def render(self, mode='human'):
        if mode == 'human':
            outfile = StringIO() if mode == 'ansi' else sys.stdout
            img = [[self._color[b] for b in line] for line in four_room_map]
            img[self._state[0]][self._state[1]] = utils.colorize(self._fill_char, "red", highlight=True)
            for line in img:
                outfile.write(f'{"".join(line)}\n')
            outfile.write('\n')

    def _next(self, action, x, y):

        def move(current_x, current_y, next_x, next_y):
            if self._grid[next_x, next_y] == BlockType.Wall.value:
                return current_x, current_y
            return next_x, next_y

        switcher = {
            Action.DOWN: lambda x, y: move(x, y, x - 1, y),
            Action.RIGHT: lambda x, y: move(x, y, x, y + 1),
            Action.UP: lambda x, y: move(x, y, x + 1, y),
            Action.LEFT: lambda x, y: move(x, y, x, y - 1),
        }
        move_func = switcher.get(action)
        return move_func(x, y)


if __name__ == "__main__":
    env = FourRoomGridWorld()
    env.render()
    env.step(Action.RIGHT)
    env.render()
    env.step(Action.RIGHT)
    env.render()
    env.step(Action.RIGHT)
    env.render()
    env.step(Action.RIGHT)
    env.render()
    env.step(Action.RIGHT)
    env.render()
    env.step(Action.RIGHT)
    env.render()
