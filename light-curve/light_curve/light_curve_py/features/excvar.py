import numpy as np

from ._base import BaseFeature


class ExcessVariance(BaseFeature):
    def _eval(self, t, m, sigma=None):
        m_mean = np.mean(m)
        d_mean = np.mean(np.power(sigma, 2))
        m_std = np.std(m, ddof=1)
        return (m_std ** 2 - d_mean) / m_mean ** 2

    @property
    def size(self):
        return 1


__all__ = ("ExcessVariance",)
