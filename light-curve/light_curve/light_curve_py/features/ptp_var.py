import numpy as np

from ._base import BaseSingleBandFeature


class PeakToPeakVar(BaseSingleBandFeature):
    r"""Peak-to-peak variation
 
    $$
    \frac{(m_i - \sigma_i)_\text{max} - (m_i + \sigma_i)_\text{min}}{(m_i - \sigma_i)_\text{max} + (m_i + \sigma_i)_\text{min}}
    $$
    For non-variable data, it should be close to zero.
    If data is close to be variable, the index should be more or equal than 0.10-0.15.
    It is sensitive to magnitude of error values and can be negative in overestimated errors case.
    
    - Depends on: **flux density**, **errors**
    - Minimum number of observations: **2**
    - Number of features: **1**

    Aller M.F., Aller H.D., Hughes P.A. 1992. [DOI:10.1086/171898](https://www.doi.org/10.1086/171898)
    """

    def _eval_single_band(self, t, m, sigma=None):
        a = np.max(m - sigma)
        b = np.min(m + sigma)
        return (a - b) / (a + b)

    @property
    def size_single_band(self):
        return 1


__all__ = ("PeakToPeakVar",)
