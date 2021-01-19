from Algorithms.BaseTD import BaseTD


class TD(BaseTD):
    def learn_single_policy(self, s, s_p, r, is_terminal):
        delta, alpha, *_ = super().learn_single_policy(s, s_p, r, is_terminal)
        self.w += alpha * delta * self.z

    def learn_multiple_policies(self, s, s_p, r, is_terminal):
        delta, alpha_vec, *_, rho, stacked_x = super().learn_multiple_policies(s, s_p, r, is_terminal)
        self.z = rho[:, None] * (self.lmbda * self.z * self.gamma_vec_t[:, None] + stacked_x)
        self.w += (alpha_vec * delta)[:, None] * self.z
        self.gamma_vec_t = self.gamma_vec_tp
