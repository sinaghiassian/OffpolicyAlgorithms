import numpy as np
from numpy.linalg import pinv
from Tasks.BaseTask import BaseTask
from Algorithms.BaseTD import BaseTD


class BaseLS(BaseTD):
    def __init__(self, task: BaseTask, **kwargs):
        super(BaseLS, self).__init__(task, **kwargs)
        self.A = np.zeros((self.task.num_features, self.task.num_features))
        self.b = np.zeros(self.task.num_features)
        self.t = 0
        if self.task.num_policies > 1:
            self.A = np.zeros((self.task.num_policies, self.task.num_features, self.task.num_features))
            self.b = np.zeros((self.task.num_policies, self.task.num_features))
            self.gamma_vec_t = np.concatenate((np.ones(2), np.zeros(6))) * self.gamma
            self.t = np.zeros(self.task.num_policies)

    def learn_single_policy(self, s, s_p, r, is_terminal):
        x, x_p = self.get_features(s, s_p, is_terminal)
        self.t += 1
        self.A += (np.outer(self.z, (x - self.gamma * x_p)) - self.A) / self.t
        self.b += (r * self.z - self.b) / self.t
        self.w = np.dot(pinv(self.A), self.b)

    def learn_multiple_policies(self, s, s_p, r, is_terminal):
        _, _, x, x_p, _, _, _, stacked_x = \
            super(BaseLS, self).learn_multiple_policies(s, s_p, r, is_terminal)
        for i in range(self.task.num_policies):
            if self.gamma_vec_t[i] != 0.0:
                self.t[i] += 1
                z = self.z[i, :]
                self.A[i, :, :] += (np.outer(z, (x - self.gamma_vec_tp[i] * x_p)) - self.A[i, :, :]) / self.t[i]
                self.b[i, :] += (self.r_vec[i] * z - self.b[i, :]) / self.t[i]
                self.w[i, :] = np.dot(pinv(self.A[i, :, :]), self.b[i, :])
        self.gamma_vec_t = self.gamma_vec_tp
