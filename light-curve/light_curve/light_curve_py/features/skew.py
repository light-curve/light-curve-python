from ._base import BaseSingleBandFeature


class Skew(BaseSingleBandFeature):
    def _eval_single_band(self, t, m, sigma=None):
        try:
            from scipy.stats import skew
        except ImportError:
            raise ImportError("scipy is required for Skew feature, please install it")

        return skew(m, bias=False)

    @property
    def size_single_band(self):
        return 1


__all__ = ("Skew",)
