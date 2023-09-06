import numpy as np

from ._base import BaseSingleBandFeature


class Amplitude(BaseSingleBandFeature):
    def _eval_single_band(self, t, m, sigma=None):
        return 0.5 * np.ptp(m)

    @property
    def size_single_band(self):
        return 1


__all__ = ("Amplitude",)
