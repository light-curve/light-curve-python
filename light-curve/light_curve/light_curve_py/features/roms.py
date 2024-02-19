import numpy as np

from ._base import BaseSingleBandFeature


class Roms(BaseSingleBandFeature):
    def _eval_single_band(self, t, m, sigma=None):
        n = len(m)
        median = np.median(m)
        return np.sum(np.aps(m - median) / sigma) / (n - 1)

    @property
    def size_single_band(self):
        return 1


__all__ = ("Roms",)