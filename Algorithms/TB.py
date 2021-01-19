from Algorithms.BaseVariableLmbda import BaseVariableLmbda


class TB(BaseVariableLmbda):
    def learn_single_policy(self, s, s_p, r, is_terminal):
        delta, alpha, x, *_, pi, _ = super().learn_single_policy(s, s_p, r, is_terminal)
        self.z = self.gamma * self.lmbda * self.old_pi * self.z + x
        self.w = self.w + alpha * delta * self.z
        self.old_pi = pi

    def learn_multiple_policies(self, s, s_p, r, is_terminal):
        delta, alpha_vec, x, x_p, pi, mu, rho, stacked_x = super().learn_multiple_policies(s, s_p, r, is_terminal)
        delta = rho * delta
        self.z = (self.gamma_vec_t * self.lmbda * self.old_pi)[:, None] * self.z + stacked_x
        self.w += alpha_vec[:, None] * (delta[:, None] * self.z)
        self.old_pi = pi
        self.gamma_vec_t = self.gamma_vec_tp
