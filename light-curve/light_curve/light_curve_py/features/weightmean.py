import numpy as np

from ._base import BaseFeature


class WeightedMean(BaseFeature):
    def _eval(self, t, m, sigma=None):
        return np.average(m, weights=np.power(sigma, -2))

    @property
    def size(self):
        return 1


__all__ = ("WeightedMean",)
