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

    def learn_single_policy(self, s, s_p, r, is_terminal):
        x, x_p = self.get_features(s, s_p, is_terminal)
        self.t += 1
        self.A += (np.dot(self.z[:, None], (x - self.gamma * x_p)[:, None]) - self.A) / self.t
        self.b += (r * self.z - self.b) / self.t
        self.w = np.dot(pinv(self.A), self.b)


class LSTD:
    def learn(self, phi, phiPrime, r, gamma, policyParameters):
        rho = policyParameters['rho']

        self.e = rho * (self.gamma_t * self.lmbda * self.e + phi)

        self.A = self.A + (np.dot(self.e[:, None], (phi - gamma * phiPrime)[None, :]) - self.A) / (self.timeStep + 1)
        self.b = self.b + (r * self.e - self.b) / (self.timeStep + 1)

        self.theta = np.dot(pinv(self.A), self.b)

        self.gamma_t = gamma
        self.timeStep = self.timeStep + 1

    def episodeInit(self):
        self.e = np.zeros(self.featureSize)
