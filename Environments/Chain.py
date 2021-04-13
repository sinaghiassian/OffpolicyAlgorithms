import numpy as np

"""A set of common utilities used within the environments. These are
not intended as API functions, and will not remain stable over time.
"""

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    """Return string surrounded by appropriate terminal color codes to
    print colorized text.  Valid colors: gray, red, green, yellow,
    blue, magenta, cyan, white, crimson
    """

    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    attrs = ';'.join(attr)
    return '\x1b[%sm%s\x1b[0m' % (attrs, string)


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
        import sys
        if mode == 'human':
            corridor_map = [
                str(i) if i > self._start_state_number
                else colorize(str(i), "blue", highlight=False)
                for i in range(self._states_number)
            ]
            corridor_map.append(colorize("T", "red", highlight=False))
            corridor_map[self._state] = colorize(corridor_map[self._state], "green", highlight=True)

            sys.stdout.write(f'{"|".join(corridor_map)}\n')


if __name__ == '__main__':
    env = Chain()
    env.reset()
    for step in range(1, 100):
        action = np.random.randint(0, 2)
        sp, r, terminal, _ = env.step(action=action)
        env.render()
        if terminal:
            env.reset()
            print('env reset')
