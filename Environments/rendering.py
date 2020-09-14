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
        self.color_policy(img, 0)
        #self.color_policy(img, 1)
        # self.color_policy(img, 2)
        # #self.color_policy(img, 3)
        # self.color_policy(img, 4)
        # #self.color_policy(img, 5)
        # self.color_policy(img, 6)
        #self.color_policy(img, 7)

        return img

    def add_error(self, error):
        if self._max_error is None:
            self._max_error = np.abs(error)
            self._valid_state = np.array(self._max_error)
            self._valid_state[self._valid_state != 0] = 1

        self._error = np.abs(error)

    def color_policy(self, img, policy_number):
        e = self._error[policy_number]
        x = self._max_error[policy_number]

        d = (255 - np.abs(x - e) * 100) * self._valid_state[policy_number]
        d = np.nan_to_num(d).reshape(11, 11).astype(np.uint8)
        img[:, :, 1] = np.transpose(d)
