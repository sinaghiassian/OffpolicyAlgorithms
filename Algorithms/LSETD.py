from Algorithms.BaseLS import BaseLS
import numpy as np


class LSETD(BaseLS):
    def __init__(self, task, **kwargs):
        super(LSETD, self).__init__(task, **kwargs)
        self.old_rho = 0
        self.F = 1
        self.beta = kwargs['beta']
        if self.task.num_policies > 1:
            self.F = np.ones(self.task.num_policies)
            self.old_rho = np.zeros(self.task.num_policies)

    @staticmethod
    def related_parameters():
        return ['alpha', 'lmbda', 'beta']

    def learn_single_policy(self, s, s_p, r, is_terminal):
        self.F = self.beta * self.old_rho * self.F + 1
        m = self.lmbda + (1 - self.lmbda) * self.F
        x, _ = self.get_features(s, s_p, is_terminal)
        rho = self.get_isr(s)
        self.z = rho * (self.gamma * self.lmbda * self.z + x * m)
        super(LSETD, self).learn_single_policy(s, s_p, r, is_terminal)
        self.old_rho = rho

    # noinspection DuplicatedCode
    def learn_multiple_policies(self, s, s_p, r, is_terminal):
        beta_vec = self.beta * self.gamma_vec_t / self.gamma
        self.F = beta_vec * self.old_rho * self.F + np.ones(self.task.num_policies)
        m = self.lmbda * np.ones(self.task.num_policies) + (1 - self.lmbda) * self.F
        stacked_x = self.task.stacked_feature_rep[:, :, s]
        rho = self.get_isr(s)
        self.z = rho[:, None] * (self.lmbda * self.z * self.gamma_vec_t[:, None] + stacked_x * m[:, None])
        super(LSETD, self).learn_multiple_policies(s, s_p, r, is_terminal)
        self.old_rho = rho

    def reset(self):
        super().reset()
        self.F = 1
        self.old_rho = 0
        if self.task.num_policies > 1:
            self.old_rho = np.zeros(self.task.num_policies)
            self.F = np.zeros(self.task.num_policies)
