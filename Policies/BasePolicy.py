from abc import ABC, abstractmethod


class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, s):
        raise NotImplementedError

    @abstractmethod
    def get_probability(self, s, a):
        raise NotImplementedError

    @abstractmethod
    def get_possible_action(self, s):
        raise NotImplementedError