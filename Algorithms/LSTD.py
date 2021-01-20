from Algorithms.BaseLS import BaseLS


class LSTD(BaseLS):
    def learn_single_policy(self, s, s_p, r, is_terminal):
        self.z = self.get_isr(s) * (self.gamma * self.lmbda * self.z + self.get_features(s, s_p, is_terminal)[0])
        super(LSTD, self).learn_single_policy(s, s_p, r, is_terminal)
