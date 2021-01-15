from Algorithms.BaseGradient import BaseGradient
import numpy as np


class HTD(BaseGradient):
    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.z_b = np.zeros(self.task.num_features)

    def learn_single_policy(self, s, s_p, r, is_terminal):
        delta, alpha, x, x_p, _ = super().learn_single_policy(s, s_p, r, is_terminal)
        alpha_v = self.compute_second_step_size()
        self.z_b = self.gamma * self.lmbda * self.z + x
        self.w += alpha * ((delta * self.z) + (x - self.gamma * x_p) * np.dot((self.z - self.z_b), self.v))
        self.v += alpha_v * ((delta * self.z) - (x - self.gamma * x_p) * np.dot(self.v, self.z_b))

    def learn_multiple_policies(self, s, s_p, r, is_terminal):
        ...
