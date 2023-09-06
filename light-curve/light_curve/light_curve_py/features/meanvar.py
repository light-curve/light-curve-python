import numpy as np

from ._base import BaseSingleBandFeature


class MeanVariance(BaseSingleBandFeature):
    def _eval_single_band(self, t, m, sigma=None):
        return np.std(m, ddof=1) / np.mean(m)

    @property
    def size_single_band(self):
        return 1


__all__ = ("MeanVariance",)
