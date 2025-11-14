from dataclasses import dataclass

import numpy as np

from ._base import BaseSingleBandFeature


@dataclass()
class BeyondNStd(BaseSingleBandFeature):
    nstd: float = 1.0

    def _eval_single_band(self, t, m, sigma=None):
        mean = np.mean(m)
        std = np.std(m, ddof=1)
        return np.count_nonzero(np.abs(m - mean) > self.nstd * std) / len(m)

    @property
    def size_single_band(self):
        return 1


__all__ = ("BeyondNStd",)
