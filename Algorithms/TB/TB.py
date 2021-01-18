from Algorithms.BaseVariableLmbda import BaseVariableLmbda


class TB(BaseVariableLmbda):
    def learn_single_policy(self, s, s_p, r, is_terminal):
        delta, alpha, x, *_, pi, _ = super().learn_single_policy(s, s_p, r, is_terminal)
        self.z = self.gamma * self.lmbda * self.old_pi * self.z + x
        self.w = self.w + alpha * delta * self.z
        self.old_pi = pi

    def learn_multiple_policies(self, s, s_p, r, is_terminal):
        ...
