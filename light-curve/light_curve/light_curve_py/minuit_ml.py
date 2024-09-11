"""Maximum-likelihood based cost function"""

from typing import Callable, Dict, Tuple

import numpy as np
from scipy.special import erf


def logpdf(x, mu, sigma):
    # We do not need second term as it does not depend on parameters
    return -(((x - mu) / sigma) ** 2) / 2  # - np.log(np.sqrt(2*np.pi) * sigma)


def barrier(x):
    res = np.where(x > 0, 1 / x, np.inf)  # FIXME: naive barrier function

    return res


def logcdf(x):
    # TODO: faster (maybe not so accurate, as we do not need it) implementation
    # return norm.logcdf(x)

    result = np.zeros(len(x))

    idx = x < -5
    result[idx] = -x[idx] ** 2 / 2 - 1 / x[idx] ** 2 - 0.9189385336 - np.log(-x[idx])
    result[~idx] = np.log(0.5) + np.log1p(erf(x[~idx] / np.sqrt(2)))

    return result


try:
    from iminuit import Minuit
except ImportError:
    MaximumLikelihood = None
else:

    class MaximumLikelihood:
        errordef = Minuit.LIKELIHOOD

        def __init__(
            self, model: Callable, parameters: Dict[str, Tuple[float, float]], upper_mask=None, *, x, y, yerror
        ):
            self.model = model
            self.x = x
            self.y = y
            self.yerror = yerror
            self.upper_mask = upper_mask

            self._parameters = parameters
            # FIXME: here we assume the order of dict keys is the same as in parameters array
            self.limits = np.array([_ for _ in parameters.values()])
            self.limits0 = self.limits[:, 0]
            self.limits1 = self.limits[:, 1]
            self.limits_scale = self.limits[:, 1] - self.limits[:, 0]

        @property
        def ndata(self):
            return len(self.y)

        def __call__(self, *par):
            ym = self.model(self.x, *par)

            if self.upper_mask is None:
                result = -np.sum(logpdf(self.y, ym, self.yerror))
            else:
                # Measurements
                result = -np.sum(logpdf(self.y[~self.upper_mask], ym[~self.upper_mask], self.yerror[~self.upper_mask]))
                # Upper limits, Tobit model
                # https://stats.stackexchange.com/questions/49443/how-to-model-this-odd-shaped-distribution-almost-a-reverse-j
                result += -np.sum(
                    logcdf((self.y[self.upper_mask] - ym[self.upper_mask]) / self.yerror[self.upper_mask])
                )

            # Barriers around parameter ranges
            # Scale is selected so that for the most of the range it is much smaller
            # than 0.5 which corresponds to 1-sigma errors
            result += 0.0001 * np.sum(barrier((par - self.limits0) / self.limits_scale))
            result += 0.0001 * np.sum(barrier((self.limits1 - par) / self.limits_scale))

            return result


__all__ = ["MaximumLikelihood"]
