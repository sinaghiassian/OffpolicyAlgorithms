from Algorithms.BaseGradient import BaseGradient
import numpy as np


class HTD(BaseGradient):
    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.z_b = np.zeros(self.task.num_features)
        if self.task.num_policies > 1:
            self.z_b = np.zeros((self.task.num_policies, self.task.num_features))

    def learn_single_policy(self, s, s_p, r, is_terminal):
        delta, alpha, x, x_p, _ = super().learn_single_policy(s, s_p, r, is_terminal)
        alpha_v = self.compute_second_step_size()
        self.z_b = self.gamma * self.lmbda * self.z_b + x
        self.w += alpha * ((delta * self.z) + (x - self.gamma * x_p) * np.dot((self.z - self.z_b), self.v))
        self.v += alpha_v * ((delta * self.z) - (x - self.gamma * x_p) * np.dot(self.v, self.z_b))

    def learn_multiple_policies(self, s, s_p, r, is_terminal):
        delta, alpha_vec, x, x_p, *_, rho, stacked_x, stacked_x_p, alphav_vec = super().learn_multiple_policies(
            s, s_p, r, is_terminal)
        self.z = rho[:, None] * (self.lmbda * self.z * self.gamma_vec_t[:, None] + stacked_x)
        self.z_b = self.lmbda * self.z_b * self.gamma_vec_t[:, None] + stacked_x
        gamma_stacked_xp = self.gamma_vec_tp[:, None] * stacked_x_p
        delta_z = delta[:, None] * self.z
        self.w += alpha_vec[:, None] * (
                delta_z + (stacked_x - gamma_stacked_xp) * (np.sum((self.z - self.z_b) * self.v, 1))[:, None])
        self.v += alphav_vec[:, None] * (
                delta_z - (stacked_x - gamma_stacked_xp) * np.sum(self.v * self.z_b, 1)[:, None])
        # TODO: Should the last v be replaced by w?
        self.gamma_vec_t = self.gamma_vec_tp

    def reset(self):
        super().reset()
        self.z_b = np.zeros(self.task.num_features)
        if self.task.num_policies > 1:
            self.z_b = np.zeros((self.task.num_policies, self.task.num_features))
