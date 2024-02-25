import numpy as np

from ._base import BaseSingleBandFeature


class Roms(BaseSingleBandFeature):
    r"""Robust median statistic

    $$
    \frac1{N-1} \sum_{i=0}^{N-1} \frac{|m_i - \mathrm{median}(m_i)|}{\sigma_i}
    $$
    For non-variable data, it should be less than one.

    - Depends on: **magnitude**, **errors**
    - Minimum number of observations: **2**
    - Number of features: **1**

    Enoch, Brown, Burgasser 2003. [DOI:10.1086/376598](https://www.doi.org/10.1086/376598)
    """

    def _eval_single_band(self, t, m, sigma=None):
        n = len(m)
        median = np.median(m)
        return np.sum(np.abs(m - median) / sigma) / (n - 1)

    @property
    def size_single_band(self):
        return 1


__all__ = ("Roms",)
