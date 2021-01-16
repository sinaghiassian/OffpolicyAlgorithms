from abc import ABC
import numpy as np
from Algorithms.BaseTD import BaseTD
from Tasks.BaseTask import BaseTask


class BaseGradient(BaseTD, ABC):
    def __init__(self, task: BaseTask, **kwargs):
        super().__init__(task, **kwargs)
        self.v = np.zeros(self.task.num_features)
        self.alpha_v = kwargs.get('alpha_v') * self.compute_step_size()

    def compute_second_step_size(self):
        return self.alpha_v
