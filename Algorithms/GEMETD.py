from Algorithms.BaseTD import BaseTD
import numpy as np


class GEMETD(BaseTD):
    """
    An ETD(0) implementation that uses GEM (aka GTD2(0) with x and x_p switched) to estimate emphasis.
    """
    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.beta = self.task.GAMMA
        self.gem_alpha = kwargs['gem_alpha']  # Step size for GEM weights.
        self.gem_beta = kwargs['gem_beta']  # Regularization parameter for GEM; not needed for a fixed target policy.
        self.k = np.zeros(self.task.num_features)  # Auxiliary weights for GEM.
        self.u = np.zeros(self.task.num_features)  # Main weights for GEM.
        if self.task.num_policies > 1:
            self.k = np.zeros((self.task.num_policies, self.task.num_features))
            self.u = np.zeros((self.task.num_policies, self.task.num_features))

    @staticmethod
    def related_parameters():
        return ['alpha', 'gem_alpha', 'gem_beta']

    def learn_single_policy(self, s, s_p, r, is_terminal):
        x, x_p = self.get_features(s, s_p, is_terminal)
        rho = self.get_isr(s)
        delta_bar = 1 + rho * self.gamma * np.dot(self.u, x) - np.dot(self.u, x_p)
        self.k += self.gem_alpha * (delta_bar - np.dot(self.k, x_p)) * x_p
        self.u += self.gem_alpha * ((x_p - self.gamma * rho * x) * np.dot(self.k, x_p) - self.gem_beta * self.u)
        delta = self.get_delta(r, x, x_p)
        m = np.dot(self.u, x)  # Use parametric estimate of expected emphasis.
        self.w += self.alpha * m * rho * delta * x

    def learn_multiple_policies(self, s, s_p, r, is_terminal):
        delta, alpha_vec, x, x_p, *_, rho, stacked_x = super().learn_multiple_policies(s, s_p, r, is_terminal)
        stacked_x_p = self.task.stacked_feature_rep[:, :, s_p]
        # GEM update:
        gem_alpha_vec = self.task.get_active_policies(s) * self.gem_alpha
        delta_bar = np.ones(self.task.num_policies) + rho * self.gamma_vec_t * np.dot(self.u, x) - np.dot(self.u, x_p)
        self.k += gem_alpha_vec[:, None] * (delta_bar[:, None] - np.sum(x_p * self.k, 1)[:, None]) * stacked_x_p
        self.u += gem_alpha_vec[:, None] * ((stacked_x_p - self.gamma_vec_t[:, None] * rho[:, None] * stacked_x) * np.sum(x_p * self.k, 1)[:, None] - self.gem_beta * self.u)  # should self.gem_beta be a vector here?
        # ETD(0) update:
        m = np.dot(self.u, x)
        self.w += (alpha_vec * m * rho * delta)[:, None] * stacked_x
        self.gamma_vec_t = self.gamma_vec_tp

    def reset(self):
        super().reset()
        self.k = np.zeros(self.task.num_features)
        self.u = np.zeros(self.task.num_features)
        if self.task.num_policies > 1:
            self.k = np.zeros((self.task.num_policies, self.task.num_features))
            self.u = np.zeros((self.task.num_policies, self.task.num_features))
