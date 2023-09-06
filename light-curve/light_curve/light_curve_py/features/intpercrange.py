from dataclasses import dataclass

from scipy.stats.mstats import mquantiles

from ._base import BaseSingleBandFeature
from ..dataclass_field import dataclass_field


@dataclass()
class InterPercentileRange(BaseSingleBandFeature):
    quantile: float = dataclass_field(default=0.25, kw_only=True)

    def _eval_single_band(self, t, m, sigma=None):
        q1, q2 = mquantiles(m, [self.quantile, 1 - self.quantile], alphap=0.5, betap=0.5)
        return q2 - q1

    @property
    def size_single_band(self):
        return 1


__all__ = ("InterPercentileRange",)
