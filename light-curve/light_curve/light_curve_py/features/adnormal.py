from scipy.stats import anderson

from ._base import BaseSingleBandFeature


class AndersonDarlingNormal(BaseSingleBandFeature):
    def _eval_single_band(self, t, m, sigma=None):
        n = len(m)
        return anderson(m).statistic * (1 + 4 / n - 25 / n**2)

    @property
    def size_single_band(self):
        return 1


__all__ = ("AndersonDarlingNormal",)
