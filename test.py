import time

import utils
from Environments.Chain import Chain
from Environments.FourRoomGridWorld import FourRoomGridWorld
from Tasks.LearnEightPoliciesTileCodingFeat import LearnEightPoliciesTileCodingFeat
import pyglet
from skimage.transform import resize
import numpy as np

from data_presister import DataPersister, find_best_performance

# if __name__ == "__main__":
# render_mode = 'human'
# render_mode = 'rgb'
# render_mode = 'screen'
#
# frames = []
# env = FourRoomGridWorld()
# # env = Chain()
# env.reset()
# actions = [2, 2, 0, 0, 0, 3, 3, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 3, 1, 1, 2, 2, 2, 0, 0, 2, 2, 2, 2,
#            2, 1, 1, 1, 1, 1, 1, 1
#            , 2, 2, 2, 3, 1, 1, 3, 3, 3, 3, 3, 0, 3, 3, 1, 3, 3, 3, 3]
# actions = actions * 1
# for step in range(len(actions)):
#     a = actions[step]
#     next_state, r, is_terminal, info = env.step(a)
#     state = next_state
#     frames.append(env.render(mode=render_mode))
#     if is_terminal:
#         env.reset()
# utils.generate_gif(frames, 'Assets/FourRoomGridWorld.gif', size=(180, 180, 3), duration=1 / 20)

# DataPersister.save_best_pref_over_first_param(exp_name="FirstChain", alg_name="HTD", auc_or_final="auc")

find_best_performance(exp_name="FirstChain", alg_name="HTD", auc_or_final="auc", second_param=0.2)
