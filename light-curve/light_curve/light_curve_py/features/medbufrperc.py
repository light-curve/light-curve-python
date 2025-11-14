from dataclasses import dataclass

import numpy as np

from ..dataclass_field import dataclass_field
from ._base import BaseSingleBandFeature


@dataclass()
class MedianBufferRangePercentage(BaseSingleBandFeature):
    quantile: float = dataclass_field(default=0.1, kw_only=True)

    def _eval_single_band(self, t, m, sigma=None):
        median = np.median(m)
        return np.count_nonzero(np.abs(median - m) < self.quantile * (np.max(m) - np.min(m)) / 2) / len(m)

    @property
    def size_single_band(self):
        return 1


__all__ = ("MedianBufferRangePercentage",)
