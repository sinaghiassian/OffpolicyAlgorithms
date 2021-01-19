from abc import ABC
import numpy as np
from Algorithms.BaseTD import BaseTD
from Tasks.BaseTask import BaseTask


class BaseGradient(BaseTD, ABC):
    def __init__(self, task: BaseTask, **kwargs):
        super().__init__(task, **kwargs)
        self.v = np.zeros(self.task.num_features)
        if self.task.num_policies > 1:
            self.v = np.zeros((self.task.num_policies, self.task.num_features))
        self.alpha_v = kwargs.get('alpha_v')

    def compute_second_step_size(self):
        return self.alpha_v * self.compute_step_size()

    def learn_multiple_policies(self, s, s_p, r, is_terminal):
        return *super().learn_multiple_policies(s, s_p, r, is_terminal), \
               self.task.stacked_feature_rep[:, :, s_p], \
               self.compute_second_step_size() * self.gamma_vec_t / self.gamma
