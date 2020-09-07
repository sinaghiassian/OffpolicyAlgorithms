import numpy as np

from Problems.BaseProblem import BaseProblem


class BaseAgent:
    def __init__(self, problem: BaseProblem, **kwargs):
        self.problem = problem
        self.w = np.zeros(self.problem.num_features)
        self.z = np.zeros(self.problem.num_features)
        if self.problem.num_policies > 1:
            self.w = np.zeros((self.problem.num_policies, self.problem.num_features))
            self.z = np.zeros((self.problem.num_policies, self.problem.num_features))
        self.gamma = kwargs['GAMMA']
        self.alpha = kwargs['alpha']
        self.lmbda = kwargs['lmbda']
        self.state_values = self.problem.load_state_values()  # This is of size num_policies * 121
        self.d_mu = self.problem.load_behavior_dist()  # same size as state_values
        self.state, self.next_state, self.action = None, None, None
        self.r_vec = np.zeros(self.problem.num_policies)
        self.gamma_vec_tp = np.zeros(self.problem.num_policies)
        self.gamma_vec_t = np.zeros(self.problem.num_policies)

    def compute_rmsve(self):
        est_value = np.dot(self.w, self.problem.feature_rep.T)
        error = est_value - self.state_values
        error_squared = error * error
        return np.sqrt(np.sum(self.d_mu * error_squared.T, 0) / np.sum(self.d_mu, 0))

    def compute_rmsve_old(self):
        est_value = np.dot(self.problem.feature_rep[:-1, :], self.w)
        error = (est_value - self.state_values[:-1])
        error_squared = error * error
        return np.sqrt(np.sum(self.d_mu[:-1] * error_squared))

    def compute_step_size(self):
        return self.alpha

    def choose_behavior_action(self):
        return self.problem.select_behavior_action(self.state)

    def choose_target_action(self):
        return self.problem.select_target_action(self.state)

    def learn(self, s, s_p, r, is_terminal):
        if self.problem.num_policies == 1:
            self.learn_single_policy(s, s_p, r, is_terminal)
        else:
            self.learn_multiple_policies(s, s_p, r, is_terminal)

    def learn_single_policy(self, s, s_p, r, is_terminal):
        raise NotImplementedError

    def learn_multiple_policies(self, s, s_p, r, is_terminal):
        raise NotImplementedError

    def __str__(self):
        return f'agent:{type(self).__name__}'
