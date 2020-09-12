import time

import utils
from Environments.FourRoomGridWorld import FourRoomGridWorld
from Problems.LearnEightPoliciesTileCodingFeat import LearnEightPoliciesTileCodingFeat
import pyglet
from skimage.transform import resize
import numpy as np

if __name__ == "__main__":

    window = pyglet.window.Window(512, 512)
    episode_number_label = pyglet.text.Label('Four Room Grid World',font_size=10,x=5, y=5)



    actions = {
        0: 'up',
        1: 'down',
        2: 'right',
        3: 'left',
    }
    env = FourRoomGridWorld()
    problem = LearnEightPoliciesTileCodingFeat()
    state = env.reset()
    img = env.render(mode='rgb')
    is_terminal = False
    frames = [img]
    for step in range(20000):
        # a = problem.select_target_action(state, policy_id=0)
        a = np.random.randint(0, 4)
        next_state, r, is_terminal, info = env.step(a)
        x, y, x_p, y_p, is_rand, selected_action = info.values()
        episode_number_label.text=(
            f'sept:{step}, '
            f's({state}):({x},{y}), '
            f'a:{actions[a]}, '
            #f'environment_action: {actions[selected_action]}, '
            f's_p({next_state}):({x_p},{y_p}), '
            #f'stochasticity:{is_rand}, '
            #f'terminal:{is_terminal}'
        )
        state = next_state
        # env.render()
        img = env.render(mode='rgb')
        frames.append(img)

        from pyglet.gl import GLubyte

        dt = resize(np.flip(img, axis=0), (window.width, window.height, 3), preserve_range=True,
                    order=0).flatten().astype(np.uint8)
        dt = (GLubyte * dt.size)(*dt)
        imageData = pyglet.image.ImageData(window.width, window.height, 'RGB', dt)
        texture = imageData.get_texture()

        window.clear()
        window.switch_to()
        window.dispatch_events()
        texture.blit(0, 0)  # draw    episode_number_label.draw()
        episode_number_label.draw()
        window.flip()

        # time.sleep(0.1)

        if is_terminal:
            break
        #utils.generate_gif(frames, 'fourRoomGridWorld.gif')
