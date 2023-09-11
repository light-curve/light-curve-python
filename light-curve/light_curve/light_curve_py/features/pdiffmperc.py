from dataclasses import dataclass

import numpy as np

from ..dataclass_field import dataclass_field
from ._base import BaseSingleBandFeature


@dataclass()
class PercentDifferenceMagnitudePercentile(BaseSingleBandFeature):
    quantile: float = dataclass_field(default=0.25, kw_only=True)

    def _eval_single_band(self, t, m, sigma=None):
        try:
            from scipy.stats.mstats import mquantiles
        except ImportError:
            raise ImportError("scipy is required for PercentDifferenceMagnitudePercentile feature, please install it")

        median = np.median(m)
        q1, q2 = mquantiles(m, [self.quantile, 1 - self.quantile], alphap=0.5, betap=0.5)
        return (q2 - q1) / median

    @property
    def size_single_band(self):
        return 1


__all__ = ("PercentDifferenceMagnitudePercentile",)
