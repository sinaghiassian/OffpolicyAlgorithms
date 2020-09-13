import time

import utils
from Environments.FourRoomGridWorld import FourRoomGridWorld
from Problems.LearnEightPoliciesTileCodingFeat import LearnEightPoliciesTileCodingFeat
import pyglet
from skimage.transform import resize
import numpy as np

if __name__ == "__main__":
    render_mode = 'human'
    render_mode = 'rgb'
    render_mode = 'screen'
    actions = {
        0: 'up',
        1: 'down',
        2: 'right',
        3: 'left',
    }
    env = FourRoomGridWorld()
    problem = LearnEightPoliciesTileCodingFeat()
    state = env.reset()
    env.render(mode=render_mode)
    is_terminal = False
    for step in range(20000):
        # a = problem.select_target_action(state, policy_id=0)
        a = np.random.randint(0, 4)
        next_state, r, is_terminal, info = env.step(a)
        x, y, x_p, y_p, is_rand, selected_action = info.values()
        print(
            f'sept:{step}, '
            f's({state}):({x},{y}), '
            f'a:{actions[a]}, '
            f'environment_action: {actions[selected_action]}, '
            f's_p({next_state}):({x_p},{y_p}), '
            f'stochasticity:{is_rand}, '
            f'terminal:{is_terminal}'
        )
        state = next_state
        env.render(mode=render_mode)
        if is_terminal:
            break
        # utils.generate_gif(frames, 'fourRoomGridWorld.gif')
