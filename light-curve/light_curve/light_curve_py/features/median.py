import numpy as np

from ._base import BaseFeature


class Median(BaseFeature):
    def _eval(self, t, m, sigma=None):
        return np.median(m)

    @property
    def size(self):
        return 1


__all__ = ("Median",)
