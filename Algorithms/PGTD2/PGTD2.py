from Algorithms.BaseGradient import BaseGradient
import numpy as np


class PGTD2(BaseGradient):
    def learn_single_policy(self, s, s_p, r, is_terminal):
        delta, alpha, x, x_p, _ = super().learn_single_policy(s, s_p, r, is_terminal)
        alpha_v = self.compute_second_step_size()
        v_mid = self.v + alpha_v * (delta * self.z - np.dot(x, self.v) * x)
        w_mid = self.w + alpha * (np.dot(x, self.v) * x - (1 - self.lmbda) * self.gamma * np.dot(self.z, self.v) * x_p)
        delta_mid = r + self.gamma * np.dot(w_mid, x_p) - np.dot(w_mid, x)
        self.w += alpha * (np.dot(x, v_mid) * x - self.gamma * (1 - self.lmbda) * np.dot(self.z, v_mid) * x_p)
        self.v += alpha_v * (delta_mid * self.z - np.dot(x, v_mid) * x)

    def learn_multiple_policies(self, s, s_p, r, is_terminal):
        ...
