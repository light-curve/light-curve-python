import numpy as np

from ._base import BaseSingleBandFeature


class WeightedMean(BaseSingleBandFeature):
    def _eval_single_band(self, t, m, sigma=None):
        return np.average(m, weights=np.power(sigma, -2))

    @property
    def size_single_band(self):
        return 1


__all__ = ("WeightedMean",)
