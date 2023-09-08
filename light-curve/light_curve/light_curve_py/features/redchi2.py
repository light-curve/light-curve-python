import numpy as np

from ._base import BaseSingleBandFeature


class ReducedChi2(BaseSingleBandFeature):
    def _eval_single_band(self, t, m, sigma=None):
        n = len(m)
        weights = sigma**-2
        m_wmean = np.average(m, weights=weights)
        s = (m - m_wmean) ** 2 * weights
        return np.sum(s) / (n - 1)

    @property
    def size_single_band(self):
        return 1


__all__ = ("ReducedChi2",)
