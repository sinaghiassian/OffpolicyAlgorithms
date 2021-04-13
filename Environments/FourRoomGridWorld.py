import numpy as np
# from Environments.rendering import Render
# from gym import utils
# import gym
# import sys

BLOCK_NORMAL, BLOCK_WALL, BLOCK_HALLWAY, BLOCK_AGENT = 0, 1, 2, 3
RGB_COLORS = {
    'red': np.array([240, 52, 52]),
    'black': np.array([0, 0, 0]),
    'green': np.array([77, 181, 33]),
    'blue': np.array([29, 111, 219]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([217, 213, 104]),
    'grey': np.array([192, 195, 196]),
    'light_grey': np.array([230, 230, 230]),
    'white': np.array([255, 255, 255])
}
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


class FourRoomGridWorld:
    def __init__(self, stochasticity_fraction=0.0):
        self._grid = np.transpose(np.flip(np.array(four_room_map, dtype=np.uint8), axis=0)[1:-1, 1:-1])
        self._max_row, self._max_col = self._grid.shape
        self._normal_tiles = np.where(self._grid == BLOCK_NORMAL)
        self._hallways_tiles = np.where(self._grid == BLOCK_HALLWAY)
        self._walls_tiles = np.where(self._grid == BLOCK_WALL)
        self.num_states = self._grid.size

        self._state = None
        self.ACTION_UP, self.ACTION_DOWN, self.ACTION_RIGHT, self.ACTION_LEFT = 0, 1, 2, 3
        self.num_actions = 4
        self._stochasticity_fraction = stochasticity_fraction
        self.hallways = {
            0: (5, 1),
            1: (1, 5),
            2: (5, 8),
            3: (8, 4)
        }
        self._window, self._info = None, None

    def reset(self):
        self._state = (0, 0)
        return self.get_state_index(*self._state)

    def step(self, action):
        x, y = self._state
        is_stochastic_selected = False
        # if self._stochasticity_fraction >= np.random.uniform():
        #     action_probability = [1 / (self.num_actions - 1) if i != action else 0 for i in range(self.num_actions)]
        #     action = np.random.choice(self.num_actions, 1, p=action_probability)[0]
        #     is_stochastic_selected = True
        x_p, y_p = self._next(action, *self._state)
        is_done = self._grid[x_p, y_p] == BLOCK_HALLWAY
        reward = 1 if is_done else 0
        self._state = (x_p, y_p)
        return self.get_state_index(*self._state), reward, False, {
            'x': x, 'y': y,
            'x_p': x_p, 'y_p': y_p,
            'is_stochastic_selected': is_stochastic_selected,
            'selected_action': action}

    def get_xy(self, state):
        return (state % self._max_row), (state // self._max_col)

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
            self.ACTION_DOWN: lambda pox_x, pos_y: move(pox_x, pos_y, pox_x, pos_y - 1),
            self.ACTION_RIGHT: lambda pox_x, pos_y: move(pox_x, pos_y, pox_x + 1, pos_y),
            self.ACTION_UP: lambda pox_x, pos_y: move(pox_x, y, pox_x, pos_y + 1),
            self.ACTION_LEFT: lambda pox_x, pos_y: move(pox_x, pos_y, pox_x - 1, pos_y),
        }
        move_func = switcher.get(action)
        return move_func(x, y)

    def render(self, mode='human'):
        import sys
        from Environments.utils import colorize
        color = {
            BLOCK_NORMAL: lambda c: colorize(c, "white", highlight=True),
            BLOCK_WALL: lambda c: colorize(c, "gray", highlight=True),
            BLOCK_HALLWAY: lambda c: colorize(c, "green", highlight=True),
        }
        if mode == 'human':
            outfile = sys.stdout
            img = [
                [color[b]('  ')
                 for x, b
                 in enumerate(line)]
                for y, line in enumerate(four_room_map)]
            img[self._max_row - self._state[1]][self._state[0] + 1] = colorize('  ', "red",
                                                                                     highlight=True)
            for line in img:
                outfile.write(f'{"".join(line)}\n')
            outfile.write('\n')
        if mode == "rgb" or mode == "screen":
            x, y = self._state
            img = np.zeros((*self._grid.shape, 3), dtype=np.uint8)
            img[self._normal_tiles] = RGB_COLORS['light_grey']

            # if render_cls is not None:
            #     assert render_cls is not type(Render), "render_cls should be Render class"
            #     img = render_cls.render(img)

            img[self._walls_tiles] = RGB_COLORS['black']
            img[self._hallways_tiles] = RGB_COLORS['green']
            img[x, y] = RGB_COLORS['red']

            ext_img = np.zeros((self._max_row + 2, self._max_col + 2, 3), dtype=np.uint8)
            ext_img[1:-1, 1:-1] = np.transpose(img, (1, 0, 2))
            if mode == "screen":

                from pyglet.window import Window
                from pyglet.text import Label
                from pyglet.gl import GLubyte
                from pyglet.image import ImageData
                zoom = 20
                if self._window is None:
                    self._window = Window((self._max_row + 2) * zoom, (self._max_col + 2) * zoom)
                    self._info = Label('Four Room Grid World', font_size=10, x=5, y=5)
                # self._info.text = f'x: {x}, y: {y}'
                dt = np.kron(ext_img, np.ones((zoom, zoom, 1)))
                dt = (GLubyte * dt.size)(*dt.flatten().astype('uint8'))
                texture = ImageData(self._window.width, self._window.height, 'RGB', dt).get_texture()
                self._window.clear()
                self._window.switch_to()
                self._window.dispatch_events()
                texture.blit(0, 0)
                # self._info.draw()
                self._window.flip()
            return np.flip(ext_img, axis=0)


if __name__ == '__main__':
    mode = 'human'
    mode = 'screen'
    env = FourRoomGridWorld()
    env.reset()
    for step in range(1, 100):
        action = np.random.randint(0, 4)
        sp, r, terminal, _ = env.step(action=action)
        env.render(mode=mode)
        if terminal:
            env.reset()
            print('env reset')
