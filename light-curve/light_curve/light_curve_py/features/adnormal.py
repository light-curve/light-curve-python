import numpy as np
from scipy.stats import anderson

from ._base import BaseFeature


class AndersonDarlingNormal(BaseFeature):
    def _eval(self, t, m, sigma=None):
        n = len(m)
        return anderson(m).statistic * (1 + 4 / n - 25 / n**2)

    @property
    def size(self):
        return 1


__all__ = ("AndersonDarlingNormal",)
