from dataclasses import dataclass

import numpy as np

from ._base import BaseFeature


@dataclass()
class BeyondNStd(BaseFeature):
    nstd: float = 1.0

    def _eval(self, t, m, sigma=None):
        mean = np.mean(m)
        std = np.std(m, ddof=1)
        return np.count_nonzero(np.abs(m - mean) > self.nstd * std) / len(m)

    @property
    def size(self):
        return 1


__all__ = ("BeyondNStd",)
