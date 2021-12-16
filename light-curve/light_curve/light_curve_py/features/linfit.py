import numpy as np

from ._base import BaseFeature
from ._lstsq import least_squares


class LinearFit(BaseFeature):
    def _eval(self, t, m, sigma=None):
        n = len(t)

        slope, chi2 = least_squares(t, m, sigma)

        red_chi2 = chi2 / (n - 2)

        weighted_t2 = np.average(t ** 2, weights=np.power(sigma, -2))
        weighted_t = np.average(t, weights=np.power(sigma, -2)) ** 2

        sigma_sum = np.sum(1 / sigma ** 2)

        return np.array([slope, np.sqrt(1 / ((weighted_t2 - weighted_t) * sigma_sum)), red_chi2])

    @property
    def size(self):
        return 3


__all__ = ("LinearFit",)
