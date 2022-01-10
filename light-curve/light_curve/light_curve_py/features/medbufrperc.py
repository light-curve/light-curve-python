from dataclasses import dataclass

import numpy as np

from ._base import BaseFeature


@dataclass()
class MedianBufferRangePercentage(BaseFeature):
    q: float = 0.1

    def _eval(self, t, m, sigma=None):
        median = np.median(m)
        return np.count_nonzero(np.abs(median - m) < self.q * (np.max(m) - np.min(m)) / 2) / len(m)

    @property
    def size(self):
        return 1


__all__ = ("MedianBufferRangePercentage",)
