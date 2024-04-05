import numpy as np

from ._base import BaseSingleBandFeature


class PeakToPeakVar(BaseSingleBandFeature):
    r"""Peak-to-peak variation

    $$
    \frac{(m_i - \sigma_i)_\text{max} - (m_i + \sigma_i)_\text{min}}
    {(m_i - \sigma_i)_\text{max} + (m_i + \sigma_i)_\text{min}}
    $$
    Input m must be non-negative (e.g. non-differential) flux density.
    This feature is a variability detector, higher values correspond to more variable sources.

    - Depends on: **flux density**, **errors**
    - Minimum number of observations: **2**
    - Number of features: **1**

    Aller M.F., Aller H.D., Hughes P.A. 1992. [DOI:10.1086/171898](https://www.doi.org/10.1086/171898)
    """

    nstd: float = 1.0

    def _eval_single_band(self, t, m, sigma=None):
        if np.any(m < 0):
            raise ValueError("m must be non-negative")
        a = np.max(m - self.nstd * sigma)
        b = np.min(m + self.nstd * sigma)
        return (a - b) / (a + b)

    @property
    def size_single_band(self):
        return 1


__all__ = ("PeakToPeakVar",)
