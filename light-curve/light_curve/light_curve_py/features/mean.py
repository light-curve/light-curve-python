import numpy as np

from ._base import BaseFeature


class Mean(BaseFeature):
    def _eval(self, t, m, sigma=None):
        return np.mean(m)

    @property
    def size(self):
        return 1


__all__ = ("Mean",)
