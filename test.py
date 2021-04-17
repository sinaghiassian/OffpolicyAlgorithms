import time

import utils
from Environments.Chain import Chain
from Environments.FourRoomGridWorld import FourRoomGridWorld
from Tasks.LearnEightPoliciesTileCodingFeat import LearnEightPoliciesTileCodingFeat
import pyglet
from skimage.transform import resize
import numpy as np

if __name__ == "__main__":
    render_mode = 'human'
    render_mode = 'rgb'
    render_mode = 'screen'

    frames = []
    env = FourRoomGridWorld()
    env = Chain()
    env.reset()
    for step in range(100):
        a = np.random.randint(0, 2)
        next_state, r, is_terminal, info = env.step(a)
        state = next_state
        frames.append(env.render(mode=render_mode))
        if is_terminal:
            env.reset()
    utils.generate_gif(frames, 'Assets/chain.gif', size=(30, 180, 3),duration=1/10)
