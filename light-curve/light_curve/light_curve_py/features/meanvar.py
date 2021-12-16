import numpy as np

from ._base import BaseFeature


class MeanVariance(BaseFeature):
    def _eval(self, t, m, sigma=None):
        return np.std(m, ddof=1) / np.mean(m)

    @property
    def size(self):
        return 1


__all__ = ("MeanVariance",)
