from Algorithms.BaseVariableLmbda import BaseVariableLmbda
import numpy as np


class Vtrace(BaseVariableLmbda):
    def learn_single_policy(self, s, s_p, r, is_terminal):
        delta, alpha, x, *_, pi, mu = super().learn_single_policy(s, s_p, r, is_terminal)
        self.z = min(self.old_rho, 1) * self.gamma * self.lmbda * self.z + x
        self.w += alpha * delta * self.z
        self.old_rho = pi / mu

    def learn_multiple_policies(self, s, s_p, r, is_terminal):
        delta, alpha_vec, x, x_p, pi, mu, rho, stacked_x = super().learn_multiple_policies(s, s_p, r, is_terminal)
        delta = rho * delta
        truncated_old_rho = np.minimum(self.old_rho, np.ones(self.task.num_policies))
        self.z = (truncated_old_rho * self.gamma_vec_t * self.lmbda)[:, None] * self.z + stacked_x
        self.w += alpha_vec[:, None] * (delta[:, None] * self.z)
        self.old_rho = rho
        self.gamma_vec_t = self.gamma_vec_tp
