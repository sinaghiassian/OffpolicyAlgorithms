from abc import ABC, abstractmethod
import numpy as np


class Render(ABC):
    @abstractmethod
    def render(self, img):
        raise NotImplementedError


class ErrorRender(Render):
    def __init__(self, num_policies, num_steps):
        self.num_steps = num_steps
        self.num_policies = num_policies
        self._error, self._max_error = None, None

    def render(self, img):
        e = self._error[0]
        x = self._max_error[0]
        d = (e - x) * 300 * e
        d = np.nan_to_num(d).reshape(11, 11).astype(np.uint8)
        img[:, :, 1] += np.transpose(d)#((np.flip(np.transpose(d), axis=1)))

        e = self._error[2]
        x = self._max_error[2]
        d = (e - x) * 300 * e
        d = np.nan_to_num(d).reshape(11, 11).astype(np.uint8)
        img[:, :, 2] += np.transpose(d)

        e = self._error[6]
        x = self._max_error[6]
        d = (e - x) * 300 * e
        d = np.nan_to_num(d).reshape(11, 11).astype(np.uint8)
        img[:, :, 0] += np.transpose(d)
        return img

    def add_error(self, error):
        if self._max_error is None:
            self._max_error = np.abs(error)
        self._error = np.abs(error)
