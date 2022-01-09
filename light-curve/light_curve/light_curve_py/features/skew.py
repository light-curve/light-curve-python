import numpy as np
from scipy.stats import skew

from ._base import BaseFeature


class Skew(BaseFeature):
    def _eval(self, t, m, sigma=None):
        return skew(m, bias=False)

    @property
    def size(self):
        return 1


__all__ = ("Skew",)
