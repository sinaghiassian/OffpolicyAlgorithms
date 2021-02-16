from Algorithms.BaseVariableLmbda import BaseVariableLmbda
import numpy as np


class ABTD(BaseVariableLmbda):
    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        zeta = kwargs.get('zeta')
        self.old_nu = 0
        if self.task.num_policies > 1:
            self.old_nu = np.zeros(self.task.num_policies)
        si_zero = self.task.ABTD_si_zero
        si_max = self.task.ABTD_si_max
        self.si = 2 * zeta * si_zero + max(0, 2 * zeta - 1) * (si_max - 2 * si_zero)

    @staticmethod
    def related_parameters():
        return['alpha', 'zeta']

    def learn_single_policy(self, s, s_p, r, is_terminal):
        delta, alpha, x, x_p, rho, pi, mu = super().learn_single_policy(s, s_p, r, is_terminal)
        nu = min(self.si, 1.0 / max(pi, mu))
        self.z = x + self.gamma * self.old_nu * self.old_pi * self.z
        self.w += alpha * delta * self.z
        self.old_nu = nu
        self.old_pi = pi

    def learn_multiple_policies(self, s, s_p, r, is_terminal):
        delta, alpha_vec, x, x_p, pi, mu, rho, stacked_x = super().learn_multiple_policies(s, s_p, r, is_terminal)
        delta = rho * delta
        nu = self.compute_nu_for_multiple_policies(pi, mu)
        self.z = (self.gamma_vec_t * self.old_nu * self.old_pi)[:, None] * self.z + stacked_x
        self.w += alpha_vec[:, None] * (delta[:, None] * self.z)
        self.old_nu = nu
        self.old_pi = pi
        self.gamma_vec_t = self.gamma_vec_tp

    def compute_nu_for_multiple_policies(self, pi, mu):
        si_vec = np.ones(self.task.num_policies) * self.si
        max_vec = 1.0 / np.maximum.reduce([pi, mu])
        return np.minimum.reduce([max_vec, si_vec])

    def reset(self):
        super().reset()
        self.old_nu = 0
