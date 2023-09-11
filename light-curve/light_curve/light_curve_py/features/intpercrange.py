from dataclasses import dataclass

from ..dataclass_field import dataclass_field
from ._base import BaseSingleBandFeature


@dataclass()
class InterPercentileRange(BaseSingleBandFeature):
    quantile: float = dataclass_field(default=0.25, kw_only=True)

    def _eval_single_band(self, t, m, sigma=None):
        try:
            from scipy.stats.mstats import mquantiles
        except ImportError:
            raise ImportError("scipy is required for InterPercentileRange feature, please install it")

        q1, q2 = mquantiles(m, [self.quantile, 1 - self.quantile], alphap=0.5, betap=0.5)
        return q2 - q1

    @property
    def size_single_band(self):
        return 1


__all__ = ("InterPercentileRange",)
