import numpy as np

from ._base import BaseSingleBandFeature


class Median(BaseSingleBandFeature):
    def _eval_single_band(self, t, m, sigma=None):
        return np.median(m)

    @property
    def size_single_band(self):
        return 1


__all__ = ("Median",)
