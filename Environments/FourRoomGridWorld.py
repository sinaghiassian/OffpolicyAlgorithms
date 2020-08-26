import sys
from io import StringIO
import gym
import numpy as np
from gym import utils

ACTION_UP, ACTION_DOWN, ACTION_RIGHT, ACTION_LEFT = 0, 1, 2, 3
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
    def __init__(self, stochastic_fraction=2.0 / 3.0):
        self._grid = np.flip(np.array(four_room_map, dtype=np.uint8), axis=0)[1:-1, 1:-1]
        self._max_row, self._max_col = self._grid.shape
        self._normal_tiles = np.where(self._grid == BLOCK_NORMAL)
        self._state = self.reset()
        self._fill_char = '  '
        self._color = {
            BLOCK_NORMAL: utils.colorize(self._fill_char, "white", highlight=True),
            BLOCK_WALL: utils.colorize(self._fill_char, "gray", highlight=True),
            BLOCK_HALLWAY: utils.colorize(self._fill_char, "green", highlight=True),
        }

        self._stochastic_fraction = stochastic_fraction

    def reset(self):
        # if random_agent_start:
        #     rnd = np.random.choice(len(self._normal_tiles[0]))
        #     self._state = (self._normal_tiles[0][rnd], self._normal_tiles[1][rnd])
        # else:
        self._state = (0, 0)
        return self._state

    def step(self, action):
        is_random = False
        if self._stochastic_fraction <= np.random.uniform():
            action_probability = [1 / 3 if i != action else 0 for i in range(4)]
            action = np.random.choice(4, 1, p=action_probability)[0]
            is_random = True
        next_x, next_y = self._next(action, *self._state)
        is_done = self._grid[next_x, next_y] == BLOCK_HALLWAY
        reward = -1
        self._state = (next_x, next_y)
        return self._state, reward, is_done, \
               {'x': self._state[0], 'y': self._state[1],
                'is_random': is_random, 'selected_action': action}

    def render(self, mode='human'):
        if mode == 'human':
            outfile = StringIO() if mode == 'ansi' else sys.stdout
            img = [[self._color[b] for b in line] for line in four_room_map]
            img[self._max_row - self._state[0]][self._state[1] + 1] = utils.colorize(self._fill_char, "red",
                                                                                     highlight=True)
            for line in img:
                outfile.write(f'{"".join(line)}\n')
            outfile.write('\n')

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
            ACTION_DOWN: lambda x, y: move(x, y, x - 1, y),
            ACTION_RIGHT: lambda x, y: move(x, y, x, y + 1),
            ACTION_UP: lambda x, y: move(x, y, x + 1, y),
            ACTION_LEFT: lambda x, y: move(x, y, x, y - 1),
        }
        move_func = switcher.get(action)
        return move_func(x, y)


if __name__ == "__main__":
    actions = {
        ACTION_UP: 'up',
        ACTION_DOWN: 'down',
        ACTION_RIGHT: 'right',
        ACTION_LEFT: 'left',
    }
    env = FourRoomGridWorld()
    env.render()
    is_terminal = False
    state = env.reset()
    for step in range(40):
        action = np.random.randint(0, 3)
        next_state, r, is_terminal, info = env.step(action)
        x, y, is_random, selected_action = info.values()
        print(
            f'sept:{step}, '
            f'state:({state[0]},{state[1]}), '
            f'action: {actions[action]}, '
            f'selected_action: {actions[selected_action]}, '
            f'next_state:({next_state[0]},{next_state[1]}), '
            f'random:{is_random}, '
            f'terminal:{is_terminal}'
        )
        state = next_state
        env.render()
        if is_terminal:
            break
