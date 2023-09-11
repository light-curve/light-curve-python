from dataclasses import dataclass

from ..dataclass_field import dataclass_field
from ._base import BaseSingleBandFeature


@dataclass()
class MagnitudePercentageRatio(BaseSingleBandFeature):
    quantile_numerator: float = dataclass_field(default=0.4, kw_only=True)
    quantile_denominator: float = dataclass_field(default=0.05, kw_only=True)

    def _eval_single_band(self, t, m, sigma=None):
        try:
            from scipy.stats.mstats import mquantiles
        except ImportError:
            raise ImportError("scipy is required for MagnitudePercentageRatio feature, please install it")

        n1, n2 = mquantiles(m, [self.quantile_numerator, 1 - self.quantile_numerator], alphap=0.5, betap=0.5)
        d1, d2 = mquantiles(m, [self.quantile_denominator, 1 - self.quantile_denominator], alphap=0.5, betap=0.5)
        return (n2 - n1) / (d2 - d1)

    @property
    def size_single_band(self):
        return 1


__all__ = ("MagnitudePercentageRatio",)
