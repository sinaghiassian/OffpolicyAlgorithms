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
        self._error, self._max_error, self._valid_state = None, None, None

    def render(self, img):
        # self.color_policy(img, 0)
        self.color_policy(img, 1)
        # self.color_policy(img, 2)
        self.color_policy(img, 3)
        # self.color_policy(img, 4)
        self.color_policy(img, 5)
        # self.color_policy(img, 6)
        self.color_policy(img, 7)

        return img

    def add_error(self, error):
        if self._max_error is None:
            self._max_error = np.abs(error).reshape(8, 11, 11)
            self._valid_state = np.array(self._max_error)
            self._valid_state[self._valid_state != 0] = 1

        self._error = np.abs(error).reshape(8, 11, 11)

    def color_policy(self, img, policy_number):
        e = self._error[policy_number]
        x = self._max_error[policy_number]
        d = np.clip((230 * e / x), 10, 255)
        d = d * self._valid_state[policy_number]
        d = np.nan_to_num(d).astype(np.uint8).T
        d = np.repeat(d, 3).reshape(11, 11, 3)
        d[:, :, 2] = 230
        c = np.where(self._valid_state[policy_number].T == 1)
        img[c] = d[c]
        return img
