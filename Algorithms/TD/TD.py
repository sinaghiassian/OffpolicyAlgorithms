from Algorithms.BaseAgent import BaseAgent
import numpy as np


class TD(BaseAgent):
    def learn(self, transition):
        delta = transition.r + self.gamma * np.dot(self.w, transition.xp) - np.dot(self.w, transition.x)
        self.w += self.compute_step_size(transition) * delta * transition.x
