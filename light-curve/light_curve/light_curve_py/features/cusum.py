import numpy as np

from ._base import BaseSingleBandFeature


class Cusum(BaseSingleBandFeature):
    def _eval_single_band(self, t, m, sigma=None):
        m_mean = np.mean(m)
        m_std = np.std(m, ddof=1)
        m_new = np.cumsum(m - m_mean)
        result = m_new / (len(m) * m_std)
        return np.ptp(result)

    @property
    def size_single_band(self):
        return 1


__all__ = ("Cusum",)
