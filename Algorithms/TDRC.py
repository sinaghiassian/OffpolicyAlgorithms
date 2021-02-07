from Algorithms.GTD import GTD
import numpy as np


class TDRC(GTD):
    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.tdrc_beta = kwargs['tdrc_beta']

    def learn_single_policy(self, s, s_p, r, is_terminal):
        delta, alpha, x, x_p, _ = super().learn_single_policy(s, s_p, r, is_terminal)
        alpha_v = self.compute_second_step_size()
        self.v += alpha_v * (delta * self.z - np.dot(x, self.v) * x) - alpha_v * self.tdrc_beta * self.v

    def learn_multiple_policies(self, s, s_p, r, is_terminal):
        delta, alpha_vec, x, x_p, *_, rho, stacked_x, stacked_x_p, alphav_vec = super().learn_multiple_policies(
            s, s_p, r, is_terminal)
        self.v += alphav_vec[:, None] * (delta[:, None] * self.z - np.sum(
            x * self.v, 1)[:, None] * stacked_x) - alphav_vec * self.tdrc_beta * self.v
