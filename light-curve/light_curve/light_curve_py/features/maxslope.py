import numpy as np

from ._base import BaseFeature


class MaximumSlope(BaseFeature):
    def _eval(self, t, m, sigma=None):
        m_span = np.subtract(m[1:], m[:-1])
        t_span = np.subtract(t[1:], t[:-1])
        div = np.abs(np.divide(m_span, t_span))
        return np.amax(div)

    @property
    def size(self):
        return 1


__all__ = ("MaximumSlope",)
