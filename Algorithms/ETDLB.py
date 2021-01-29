from Algorithms.BaseTD import BaseTD
import numpy as np


class ETDLB(BaseTD):
    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.F = 1
        self.old_rho = 0
        self.beta = kwargs.get('beta')
        if self.task.num_policies > 1:
            self.F = np.zeros(self.task.num_policies)
            self.old_rho = np.zeros(self.task.num_policies)

    @staticmethod
    def related_parameters():
        return ['alpha', 'lmbda', 'beta']

    def learn_single_policy(self, s, s_p, r, is_terminal):
        x, x_p = self.get_features(s, s_p, is_terminal)
        delta = self.get_delta(r, x, x_p)
        self.F = self.beta * self.old_rho * self.F + 1
        m = self.lmbda * 1 + (1 - self.lmbda) * self.F
        rho = self.get_isr(s)
        self.z = rho * (x * m + self.gamma * self.lmbda * self.z)
        self.w += self.compute_step_size() * delta * self.z
        self.old_rho = rho

    def learn_multiple_policies(self, s, s_p, r, is_terminal):
        delta, alpha_vec, *_, rho, _ = super().learn_multiple_policies(s, s_p, r, is_terminal)
        stacked_x = self.task.stacked_feature_rep[:, :, s]
        beta_vec = self.beta * self.gamma_vec_t / self.gamma
        self.F = beta_vec * self.old_rho * self.F + np.ones(self.task.num_policies)
        m = self.lmbda * np.ones(self.task.num_policies) + (1 - self.lmbda) * self.F
        self.z = rho[:, None] * (self.lmbda * self.z * self.gamma_vec_t[:, None] + stacked_x * m[:, None])
        self.w += (alpha_vec * delta)[:, None] * self.z
        self.old_rho = rho
        self.gamma_vec_t = self.gamma_vec_tp

    def reset(self):
        super().reset()
        self.F = 1
        self.old_rho = 0
        if self.task.num_policies > 1:
            self.old_rho = np.zeros(self.task.num_policies)
            self.F = np.zeros(self.task.num_policies)