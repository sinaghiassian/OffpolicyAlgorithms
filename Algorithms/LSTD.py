from Algorithms.BaseLS import BaseLS


class LSTD(BaseLS):
    def learn_single_policy(self, s, s_p, r, is_terminal):
        x, _ = self.get_features(s, s_p, is_terminal)
        self.z = self.get_isr(s) * (self.gamma * self.lmbda * self.z + x)
        super(LSTD, self).learn_single_policy(s, s_p, r, is_terminal)

    def learn_multiple_policies(self, s, s_p, r, is_terminal):
        x, _ = self.get_features(s, s_p, is_terminal)
        self.z = self.get_isr(s)[:, None] * (self.lmbda * self.z * self.gamma_vec_t[:, None] + x)
        super(LSTD, self).learn_multiple_policies(s, s_p, r, is_terminal)
