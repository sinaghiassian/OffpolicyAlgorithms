from Algorithms.BaseTD import BaseTD
from Tasks.BaseTask import BaseTask
import numpy as np


class BaseVariableLmbda(BaseTD):
    def __init__(self, task: BaseTask, **kwargs):
        super().__init__(task, **kwargs)
        self.old_pi, self.old_mu = 0, 1
        if self.task.num_policies > 1:
            self.old_pi, self.old_mu = np.zeros(self.task.num_policies), np.ones(self.task.num_policies)
        self.old_rho = self.old_pi / self.old_mu

    def learn_single_policy(self, s, s_p, r, is_terminal):
        alpha = self.compute_step_size()
        pi = self.task.get_pi(s, self.action)
        mu = self.task.get_mu(s, self.action)
        rho = pi / mu
        x, x_p = self.get_features(s, s_p, is_terminal)
        delta = rho * self.get_delta(r, x, x_p)
        return delta, alpha, x, x_p, rho, pi, mu

    def reset(self):
        self.old_pi, self.old_mu = 0, 1
        self.old_rho = self.old_pi / self.old_mu
