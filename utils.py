import numpy as np
from collections import namedtuple

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


def generate_gif(frames, path):
    import imageio
    from skimage.transform import resize
    for idx, frame_idx in enumerate(frames):
        frames[idx] = resize(frame_idx, (256, 256, 3), preserve_range=True, order=0).astype(np.uint8)
    imageio.mimsave(path, frames, duration=1 / 10)
