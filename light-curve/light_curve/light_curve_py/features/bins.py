import sys
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ._base_meta import BaseMetaFeature

if sys.version_info >= (3, 10):
    from dataclasses import field
else:
    from dataclasses import field as _field

    def field(*, kw_only, **kwargs):
        return _field(**kwargs)


@dataclass()
class Bins(BaseMetaFeature):
    window: float = field(default=1.0, kw_only=True)
    offset: float = field(default=0.0, kw_only=True)

    def transform(self, t, m, sigma=None, *, sorted=None, fill_value=None):
        assert self.window > 0, "Window should be a positive number."
        n = np.ceil((t[-1] - t[0]) / self.window) + 1
        j = np.arange(0, n)
        bins = j * self.window

        delta = self.window * np.floor((t[0] - self.offset) / self.window)
        time = t - self.offset - delta

        idx = np.digitize(time, bins)
        uniq_idx, nums = np.unique(idx, return_counts=True)

        new_time = uniq_idx * self.window + self.offset - self.window / 2 + delta

        weights = np.power(sigma, -2)
        s = ndimage.sum(weights, labels=idx, index=uniq_idx)
        new_magn = ndimage.sum(m * weights, labels=idx, index=uniq_idx) / s
        new_sigma = np.sqrt(nums / s)

        return new_time, new_magn, new_sigma


__all__ = ("Bins",)
