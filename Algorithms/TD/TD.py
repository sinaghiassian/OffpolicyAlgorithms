from Algorithms.BaseAgent import BaseAgent
import numpy as np


class TD(BaseAgent):
    def learn(self, s, s_p, r):
        pi = self.problem.get_pi(s, self.action)
        mu = self.problem.get_mu(s, self.action)
        rho = pi / mu
        x_p = self.problem.get_state_feature_rep(s_p)
        x = self.problem.get_state_feature_rep(s)
        delta = rho * (r + self.gamma * np.dot(self.w, x_p) - np.dot(self.w, x))
        self.w += self.compute_step_size() * delta * x
