import numpy as np

from ._base import BaseFeature


class Kurtosis(BaseFeature):
    def _eval(self, t, m, sigma=None):
        n = len(m)
        m_mean = np.mean(m)
        m_st = np.std(m, ddof=1) ** 4
        m_sum = sum(np.power(m - m_mean, 4))
        return (n * (n + 1) * m_sum) / ((n - 1) * (n - 2) * (n - 3) * m_st) - 3 * np.power((n - 1), 2) / (
            (n - 2) * (n - 3)
        )

    @property
    def size(self):
        return 1


__all__ = ("Kurtosis",)
