from Algorithms.BaseGradient import BaseGradient
import numpy as np


class GTD2(BaseGradient):
    def learn_single_policy(self, s, s_p, r, is_terminal):
        delta, alpha, x, x_p, *_ = super().learn_single_policy(s, s_p, r, is_terminal)
        alpha_v = self.compute_second_step_size()
        self.w += alpha * (np.dot(x, self.w) * x - self.gamma * (1 - self.lmbda) * np.dot(self.z, self.w) * x_p)
        self.v += alpha_v * (delta * self.z - np.dot(x, self.v) * x)

    def learn_multiple_policies(self, s, s_p, r, is_terminal):
        ...
