from Algorithms.BaseVariableLmbda import BaseVariableLmbda


class Vtrace(BaseVariableLmbda):
    def learn_single_policy(self, s, s_p, r, is_terminal):
        delta, alpha, x, *_, pi, mu = super().learn_single_policy(s, s_p, r, is_terminal)
        self.z = min(self.old_rho, 1) * self.gamma * self.lmbda * self.z + x
        self.w += alpha * delta * self.z
        self.old_rho = pi / mu

    def learn_multiple_policies(self, s, s_p, r, is_terminal):
        ...
