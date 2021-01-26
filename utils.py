import numpy as np
from collections import namedtuple
import os

Transition = namedtuple('Transition',
                        ('x', 'a', 'xp', 'r'))


class ImmutableDict(dict):
    def immutable(self):
        raise TypeError("%r objects are immutable" % self.__class__.__name__)

    def __setitem__(self, key, value):
        self.immutable()

    def __delitem__(self, key):
        self.immutable()

    def set_default(self, k, default):
        self.immutable()

    def update(self, __m, **kwargs):
        self.immutable()

    def clear(self) -> None:
        self.immutable()


def create_name_for_saving(param_dict):
    final_str = ''
    for k, v in param_dict.items():
        if k == 'alpha' or k == 'eta':
            split_str = str.split(f'{v:.10f}', '.')
        else:
            split_str = str.split(f'{v:.5f}', '.')
        final_str += '_' + k + split_str[0] + split_str[1]
    return final_str


def save_result(path, name, result_array, params):
    name_to_save = create_name_for_saving(param_dict=params)
    path_and_name = os.path.join(path, name_to_save)
    final_name = path_and_name + name
    np.save(final_name, result_array)


def generate_gif(frames, path):
    import imageio
    from skimage.transform import resize
    for idx, frame_idx in enumerate(frames):
        frames[idx] = resize(frame_idx, (256, 256, 3), preserve_range=True, order=0).astype(np.uint8)
    imageio.mimsave(path, frames, duration=1 / 10)
