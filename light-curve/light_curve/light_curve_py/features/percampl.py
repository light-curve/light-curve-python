import numpy as np

from ._base import BaseFeature


class PercentAmplitude(BaseFeature):
    def _eval(self, t, m, sigma=None):
        median = np.median(m)
        return np.max((np.max(m) - median, median - np.min(m)))

    @property
    def size(self):
        return 1


__all__ = ("PercentAmplitude",)
