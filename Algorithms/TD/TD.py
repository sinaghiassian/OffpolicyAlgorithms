from Algorithms.BaseAgent import BaseAgent
import numpy as np


class TD(BaseAgent):
    def learn_single_policy(self, s, s_p, r, is_terminal):
        pi = self.problem.get_pi(s, self.action)
        mu = self.problem.get_mu(s, self.action)
        rho = pi / mu
        x_p = np.zeros(self.problem.num_features)
        if not is_terminal:
            x_p = self.problem.get_state_feature_rep(s_p)
        x = self.problem.get_state_feature_rep(s)
        delta = rho * (r + self.gamma * np.dot(self.w, x_p) - np.dot(self.w, x))
        alpha = self.compute_step_size()
        self.w += alpha * delta * x

    def learn_multiple_policies(self, s, s_p, r, is_terminal):
        active_policies_vec = self.problem.get_active_policies(s)
        self.r_vec = np.zeros(self.problem.num_policies)
        if r > 0:
            terminal_policies_vec = self.problem.get_terminal_policies(s_p)
            self.r_vec = r * terminal_policies_vec
        alpha_vec = active_policies_vec * self.compute_step_size()
        x = self.problem.get_state_feature_rep(s)
        x_p = np.zeros(self.problem.num_features)
        if not is_terminal:
            x_p = self.problem.get_state_feature_rep(s_p)
        stacked_x = self.problem.stacked_feature_rep[:, :, s]
        pi = self.problem.get_pi(s, self.action)
        mu = self.problem.get_mu(s, self.action)
        rho = pi / mu
        self.gamma_vec_tp = self.problem.get_active_policies(s_p) * self.gamma
        delta = self.r_vec + self.gamma_vec_tp * np.dot(self.w, x_p) - np.dot(self.w, x)
        self.z = rho[:, None] * (self.lmbda * self.z * self.gamma_vec_t[:, None] + stacked_x)
        self.w += (alpha_vec * delta)[:, None] * self.z
        self.gamma_vec_t = self.gamma_vec_tp
