from ._base import BaseSingleBandFeature


class AndersonDarlingNormal(BaseSingleBandFeature):
    def _eval_single_band(self, t, m, sigma=None):
        try:
            from scipy.stats import anderson
        except ImportError:
            raise ImportError("scipy is required for AndersonDarlingNormal feature, please install it")

        n = len(m)
        return anderson(m).statistic * (1 + 4 / n - 25 / n**2)

    @property
    def size_single_band(self):
        return 1


__all__ = ("AndersonDarlingNormal",)
