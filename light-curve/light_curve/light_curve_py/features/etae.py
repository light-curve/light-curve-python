import numpy as np

from ._base import BaseSingleBandFeature


class EtaE(BaseSingleBandFeature):
    def _eval_single_band(self, t, m, sigma=None):
        n = len(m)
        m_std = np.var(m, ddof=1)
        m_sum = np.sum(((m[1:] - m[:-1]) / (t[1:] - t[:-1])) ** 2)
        return m_sum * (t[n - 1] - t[0]) ** 2 / ((n - 1) ** 3 * m_std)

    @property
    def size_single_band(self):
        return 1


__all__ = ("EtaE",)
