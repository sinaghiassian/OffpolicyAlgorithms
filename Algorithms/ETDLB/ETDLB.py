from Algorithms.BaseTD import BaseTD
import numpy as np


class ETDLB(BaseTD):
    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.F = 1
        self.old_rho = 0
        self.beta = kwargs.get('beta')

    def learn_single_policy(self, s, s_p, r, is_terminal):
        x, x_p = self.get_features(s, s_p, is_terminal)
        delta = self.get_delta(r, x, x_p)
        self.F = self.beta * self.old_rho * self.F + 1
        m = self.lmbda * 1 + (1 - self.lmbda) * self.F
        rho = self.get_isr(s)
        self.z = rho * (x * m + self.gamma * self.lmbda * self.z)
        self.w += self.compute_step_size() * delta * self.z
        self.old_rho = rho

    def reset(self):
        super().reset()
        self.F = 1
        self.old_rho = 0

    def learn_multiple_policies(self, s, s_p, r, is_terminal):
        ...
