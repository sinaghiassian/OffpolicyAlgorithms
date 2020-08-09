from Algorithms.TD.TD import TD
import numpy as np


class ITD(TD):
    def compute_step_size(self, transition):
        step_size = self.alpha / (1 + self.alpha * np.dot(
            (transition.x - self.gamma * transition.xp), transition.x) * transition.x)
        return step_size
