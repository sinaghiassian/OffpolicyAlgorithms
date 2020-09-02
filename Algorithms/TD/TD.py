from Algorithms.BaseAgent import BaseAgent, BaseAgentLearnMultiplePolicies
import numpy as np


class TD(BaseAgent):
    def learn(self, s, s_p, r):
        pi = self.problem.get_pi(s, self.action)
        mu = self.problem.get_mu(s, self.action)
        rho = pi / mu
        x_p = self.problem.get_state_feature_rep(s_p)
        x = self.problem.get_state_feature_rep(s)
        delta = rho * (r + self.gamma * np.dot(self.w, x_p) - np.dot(self.w, x))
        alpha = self.compute_step_size()
        self.w += alpha * delta * x


class TDMultiplePolicy(BaseAgentLearnMultiplePolicies):
    def learn(self, s, s_p, r):
        x = self.problem.get_state_feature_rep(s)
        x_p = self.problem.get_state_feature_rep(s_p)
        stacked_x = self.problem.stacked_feature_rep[s]
        stacked_x_p = self.problem.stacked_feature_rep[s_p]
        pi = self.problem.get_pi(s, self.action)
        mu = self.problem.get_mu(s, self.action)
        rho = pi / mu
        delta = r + rho * np.dot(self.w, x_p) - np.dot(self.w, x)
        self.z = rho[:, None] * (self.lmbda * self.z * self.gamma_vec)
        active_policies = self.problem.get_active_policies(s) * self.gamma
