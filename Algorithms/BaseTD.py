import numpy as np
from Tasks.BaseTask import BaseTask


class BaseTD:
    def __init__(self, task: BaseTask, **kwargs):
        self.task = task
        self.w = np.zeros(self.task.num_features)
        self.z = np.zeros(self.task.num_features)
        if self.task.num_policies > 1:
            self.w = np.zeros((self.task.num_policies, self.task.num_features))
            self.z = np.zeros((self.task.num_policies, self.task.num_features))
        self.gamma = kwargs['GAMMA']
        self.alpha = kwargs['alpha']
        self.lmbda = kwargs['lmbda']
        self.state_values = self.task.load_state_values()  # This is of size num_policies * 121
        self.d_mu = self.task.load_behavior_dist()  # same size as state_values
        self.state, self.next_state, self.action = None, None, None
        self.r_vec = np.zeros(self.task.num_policies)
        self.gamma_vec_tp = np.zeros(self.task.num_policies)
        self.gamma_vec_t = np.zeros(self.task.num_policies)

    def compute_rmsve(self):
        est_value = np.dot(self.w, self.task.feature_rep.T)
        error = est_value - self.state_values
        error_squared = error * error
        return np.sqrt(np.sum(self.d_mu * error_squared.T, 0) / np.sum(self.d_mu, 0)), error

    def compute_step_size(self):
        return self.alpha

    def choose_behavior_action(self):
        return self.task.select_behavior_action(self.state)

    def choose_target_action(self):
        return self.task.select_target_action(self.state)

    def learn(self, s, s_p, r, is_terminal):
        if self.task.num_policies == 1:
            self.learn_single_policy(s, s_p, r, is_terminal)
        else:
            self.learn_multiple_policies(s, s_p, r, is_terminal)

    def get_features(self, s, s_p, is_terminal):
        x_p = np.zeros(self.task.num_features)
        if not is_terminal:
            x_p = self.task.get_state_feature_rep(s_p)
        x = self.task.get_state_feature_rep(s)
        return x, x_p

    def get_isr(self, s):
        pi = self.task.get_pi(s, self.action)
        mu = self.task.get_mu(s, self.action)
        rho = pi / mu
        return rho

    def get_delta(self, r, x, x_p):
        return r + self.gamma * np.dot(self.w, x_p) - np.dot(self.w, x)

    def learn_single_policy(self, s, s_p, r, is_terminal):
        x, x_p = self.get_features(s, s_p, is_terminal)
        rho = self.get_isr(s)
        alpha = self.compute_step_size()
        delta = self.get_delta(r, x, x_p)
        self.z = rho * (self.gamma * self.lmbda * self.z + x)
        return delta, alpha, x, x_p, rho

    def learn_multiple_policies(self, s, s_p, r, is_terminal):
        raise NotImplementedError

    def reset(self):
        self.z = np.zeros(self.task.num_features)
        if self.task.num_policies > 1:
            self.z = np.zeros((self.task.num_policies, self.task.num_features))

    def __str__(self):
        return f'agent:{type(self).__name__}'
