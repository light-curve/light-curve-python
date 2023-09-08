import numpy as np

from ._base import BaseSingleBandFeature


class StetsonK(BaseSingleBandFeature):
    def _eval_single_band(self, t, m, sigma=None):
        m_mean = np.average(m, weights=np.power(sigma, -2))
        numerator = np.sum(np.abs((m - m_mean) / sigma))
        chisq = np.sum(((m - m_mean) / sigma) ** 2)
        return numerator / np.sqrt(len(m) * chisq)

    @property
    def size_single_band(self):
        return 1


__all__ = ("StetsonK",)
