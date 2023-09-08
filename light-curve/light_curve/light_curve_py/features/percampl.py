import numpy as np

from ._base import BaseSingleBandFeature


class PercentAmplitude(BaseSingleBandFeature):
    def _eval_single_band(self, t, m, sigma=None):
        median = np.median(m)
        return np.max((np.max(m) - median, median - np.min(m)))

    @property
    def size_single_band(self):
        return 1


__all__ = ("PercentAmplitude",)
