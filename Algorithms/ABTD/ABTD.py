from Algorithms.BaseVariableLmbda import BaseVariableLmbda


class ABTD(BaseVariableLmbda):
    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        zeta = kwargs.get('zeta')
        si_zero = 1
        si_max = 2
        self.old_nu = 0
        self.si = 2 * zeta * si_zero + max(0, 2 * zeta - 1) * (si_max - 2 * si_zero)

    def learn_single_policy(self, s, s_p, r, is_terminal):
        delta, alpha, x, x_p, rho, pi, mu = super().learn_single_policy(s, s_p, r, is_terminal)
        nu = min(self.si, 1.0 / max(pi, mu))
        self.z = x + self.gamma * self.old_nu * self.old_pi * self.z
        self.w += alpha * delta * self.z
        self.old_nu = nu
        self.old_pi = pi

    def reset(self):
        super().reset()
        self.old_nu = 0

    def learn_multiple_policies(self, s, s_p, r, is_terminal):
        ...
