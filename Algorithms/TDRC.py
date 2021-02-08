from Algorithms.BaseGradient import BaseGradient
import numpy as np


# noinspection DuplicatedCode
class TDRC(BaseGradient):
    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.tdrc_beta = kwargs['tdrc_beta']

    @staticmethod
    def related_parameters():
        return ['alpha', 'lmbda', 'eta', 'tdrc_beta']

    def learn_single_policy(self, s, s_p, r, is_terminal):
        delta, alpha, x, x_p, _ = super().learn_single_policy(s, s_p, r, is_terminal)
        alpha_v = self.compute_second_step_size()
        self.w += alpha * (delta * self.z - self.gamma * (1 - self.lmbda) * np.dot(self.z, self.v) * x_p)
        self.v += alpha_v * (delta * self.z - np.dot(x, self.v) * x) - alpha_v * self.tdrc_beta * self.v

    def learn_multiple_policies(self, s, s_p, r, is_terminal):
        delta, alpha_vec, x, x_p, *_, rho, stacked_x, stacked_x_p, alphav_vec = super().learn_multiple_policies(
            s, s_p, r, is_terminal)
        self.z = rho[:, None] * (self.lmbda * self.z * self.gamma_vec_t[:, None] + stacked_x)
        phi_prime_multiplier = (1 - self.lmbda) * self.gamma_vec_tp * np.sum(self.z * self.v, 1)
        self.w += alpha_vec[:, None] * (delta[:, None] * self.z - phi_prime_multiplier[:, None] * stacked_x_p)
        self.v += alphav_vec[:, None] * (delta[:, None] * self.z - np.sum(
            x * self.v, 1)[:, None] * stacked_x) - (alphav_vec * self.tdrc_beta)[:, None] * self.v
        self.gamma_vec_t = self.gamma_vec_tp
