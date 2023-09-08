import numpy as np

from ._base import BaseSingleBandFeature


class Eta(BaseSingleBandFeature):
    def _eval_single_band(self, t, m, sigma=None):
        n = len(m)
        m_std = np.var(m, ddof=1)
        m_sum = np.sum((m[1:] - m[:-1]) ** 2)
        return m_sum / ((n - 1) * m_std)

    @property
    def size_single_band(self):
        return 2


__all__ = ("Eta",)
