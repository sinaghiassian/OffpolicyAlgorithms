from Algorithms.BaseLS import BaseLS


class LSETD(BaseLS):
    def __init__(self, task, **kwargs):
        super(LSETD, self).__init__(task, **kwargs)
        self.old_rho = 0
        self.F = 1
        self.beta = kwargs['beta']

    def learn_single_policy(self, s, s_p, r, is_terminal):
        self.F = self.beta * self.old_rho * self.F + 1
        m = self.lmbda + (1 - self.lmbda) * self.F
        rho = self.get_isr(s)
        self.z = rho * (self.gamma * self.lmbda * self.z + self.get_features(s, s_p, is_terminal)[0] * m)
        super(LSETD, self).learn_single_policy(s, s_p, r, is_terminal)
