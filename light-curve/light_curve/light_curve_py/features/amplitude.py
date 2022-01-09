import numpy as np

from ._base import BaseFeature


class Amplitude(BaseFeature):
    def _eval(self, t, m, sigma=None):
        return 0.5 * np.ptp(m)

    @property
    def size(self):
        return 1


__all__ = ("Amplitude",)
