from Algorithms.ETDLB import ETDLB


class ETD(ETDLB):
    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.beta = self.task.GAMMA

    @staticmethod
    def related_parameters():
        return ['alpha', 'lmbda']
