from collections import namedtuple

Transition = namedtuple('Transition',
                        ('x', 'a', 'xp', 'r'))


class ImmutableDict(dict):
    def immutable(self):
        raise TypeError("%r objects are immutable" % self.__class__.__name__)

    def __setitem__(self, key, value):
        self.immutable()

    def __delitem__(self, key):
        self.immutable()

    def setdefault(self, k, default):
        self.immutable()

    def update(self, __m, **kwargs):
        self.immutable()

    def clear(self) -> None:
        self.immutable()
