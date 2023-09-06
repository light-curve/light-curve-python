import numpy as np

from ._base import BaseSingleBandFeature


class StandardDeviation(BaseSingleBandFeature):
    def _eval_single_band(self, t, m, sigma=None):
        return np.std(m, ddof=1)

    @property
    def size_single_band(self):
        return 1


__all__ = ("StandardDeviation",)
