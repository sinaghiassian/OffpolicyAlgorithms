import numpy as np


class Chain:
    def __init__(self, states_number: int = 8, start_state_number: int = 4, **kwargs):
        assert start_state_number < states_number, "start states numbers should be less than state number"

        self._states_number = states_number
        self._start_state_number = start_state_number
        self._terminal = self._states_number
        self._state = None
        self.RIGHT_ACTION = 0
        self.RETREAT_ACTION = 1
        self.num_states = states_number
        self._window = None

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

    def render(self, mode='human'):
        if mode == 'human':
            import sys
            from Environments.utils import colorize
            corridor_map = [
                str(i) if i > self._start_state_number
                else colorize(str(i), "blue", highlight=False)
                for i in range(self._states_number)
            ]
            corridor_map.append(colorize("T", "red", highlight=False))
            corridor_map[self._state] = colorize(corridor_map[self._state], "green", highlight=True)

            sys.stdout.write(f'{"|".join(corridor_map)}\n')

        if mode == "rgb" or mode == "screen":
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
            img = np.zeros((self.num_states, 1, 3), dtype=np.uint8)
            img[:, 0] = RGB_COLORS['grey']
            img[:self._start_state_number - 1, 0] = RGB_COLORS['yellow']
            img[self._terminal - 1, 0] = RGB_COLORS['black']
            img[self._state - 1, 0] = RGB_COLORS['green']

            img = np.transpose(img, (1, 0, 2))
            if mode == "screen":
                from pyglet.window import Window
                from pyglet.text import Label
                from pyglet.gl import GLubyte
                from pyglet.image import ImageData
                zoom = 50
                if self._window is None:
                    self._window = Window(self.num_states * zoom, 1 * zoom)

                dt = np.kron(img, np.ones((zoom, zoom, 1)))
                dt = (GLubyte * dt.size)(*dt.flatten().astype('uint8'))
                texture = ImageData(self._window.width, self._window.height, 'RGB', dt).get_texture()
                self._window.clear()
                self._window.switch_to()
                self._window.dispatch_events()
                texture.blit(0, 0)
                # self._info.draw()
                self._window.flip()
            return np.flip(img, axis=0)


if __name__ == '__main__':
    env = Chain()
    env.reset()
    for step in range(1, 1000):
        action = np.random.randint(0, 2)
        sp, r, terminal, _ = env.step(action=action)
        env.render(mode="screen")
        if terminal:
            env.reset()
            print('env reset')
