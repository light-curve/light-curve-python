from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from light_curve.light_curve_py.dataclass_field import dataclass_field
from light_curve.light_curve_py.warnings import mark_experimental


@dataclass
class BaseMultiBandFeature(ABC):
    @property
    @abstractmethod
    def size(self) -> int:
        pass

    @property
    @abstractmethod
    def is_band_required(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_multiband_supported(self) -> bool:
        pass

    def _eval(self, *, t, m, sigma, band):
        """It could be missed if _eval_and_fill is re-implemented"""
        raise NotImplementedError("_eval is missed")

    @abstractmethod
    def _eval_and_fill(self, *, t, m, sigma, band, fill_value):
        """It has a default implementation, but it requires _eval to be implemented"""
        try:
            a = self._eval(t=t, m=m, sigma=sigma, band=band)
            if np.any(~np.isfinite(a)):
                raise ValueError
            return a
        except (ValueError, ZeroDivisionError, RuntimeError) as e:
            if fill_value is not None:
                return np.full(self.size, fill_value)
            raise e

    @mark_experimental
    def __post_init__(self) -> None:
        pass

    def _normalize_input(self, *, t, m, sigma, band, sorted, check):
        t = np.asarray(t)
        m = np.asarray(m)

        if sigma is not None:
            sigma = np.asarray(sigma)

        if band is None and self.is_band_required:
            raise ValueError("band is required")
        if band is not None:
            if not self.is_multiband_supported:
                raise ValueError(
                    "(band != None) is not supported by this feature instance, consider to pass a band array "
                    "or recreate the feature instance with different parameters"
                )
            band = np.asarray(band)
            if band.ndim != 1:
                raise ValueError("band must be None or 1D array-like")

        if check:
            if np.any(~np.isfinite(t)):
                raise ValueError("t values must be finite")
            if np.any(~np.isfinite(m)):
                raise ValueError("m values must be finite")
            if sigma is not None and np.any(np.isnan(sigma)):
                raise ValueError("sigma must have no NaNs")

        if sorted is None:
            diff = np.diff(t)
            if np.any(diff == 0):
                raise ValueError("t must be unique")
            if np.any(diff < 0):
                raise ValueError("t must be sorted")
        elif not sorted:
            idx = np.argsort(t)
            t = t[idx]
            m = m[idx]
            if sigma is not None:
                sigma = sigma[idx]
            if band is not None:
                band = band[idx]

        return t, m, sigma, band

    def __call__(self, t, m, sigma=None, band=None, *, sorted=None, check=True, fill_value=None):
        t, m, sigma, band = self._normalize_input(t=t, m=m, sigma=sigma, band=band, sorted=sorted, check=check)
        return self._eval_and_fill(t=t, m=m, sigma=sigma, band=band, fill_value=fill_value)

    def many(self, lcs, *, sorted=None, check=True, fill_value=None, n_jobs=-1):
        """Extract features in bulk

        This exists for computability only and doesn't support parallel
        execution, that's why `n_jobs=1` must be used
        """
        if n_jobs != 1:
            raise NotImplementedError("Parallel execution is not supported by this feature, use n_jobs=1")
        return np.stack([self(*lc, sorted=sorted, check=check, fill_value=fill_value) for lc in lcs])


@dataclass
class BaseSingleBandFeature(BaseMultiBandFeature):
    bands: Optional[Sequence[str]] = dataclass_field(default=None, kw_only=True)

    @property
    @abstractmethod
    def size_single_band(self) -> int:
        pass

    @abstractmethod
    def _eval_single_band(self, *, t, m, sigma):
        pass

    @property
    def n_bands(self) -> int:
        if self.bands is None:
            return 1
        return len(self.bands)

    @property
    def size(self) -> int:
        return self.n_bands * self.size_single_band

    @property
    def is_band_required(self) -> bool:
        return self.bands is not None

    @property
    def is_multiband_supported(self) -> bool:
        return self.bands is not None

    def _eval_and_fill_single_band(self, *, t, m, sigma, fill_value):
        try:
            a = self._eval_single_band(t=t, m=m, sigma=sigma)
            if np.any(~np.isfinite(a)):
                raise ValueError
            return a
        except (ValueError, ZeroDivisionError, RuntimeError) as e:
            if fill_value is not None:
                return np.full(self.size_single_band, fill_value)
            raise e

    def _eval_and_fill(self, *, t, m, sigma, band, fill_value):
        if self.bands is None:
            return self._eval_and_fill_single_band(t=t, m=m, sigma=sigma, fill_value=fill_value)

        values = []
        for band_to_calc in self.bands:
            band_mask = band == band_to_calc
            t_band = t[band_mask]
            m_band = m[band_mask]
            if sigma is None:
                sigma_band = None
            else:
                sigma_band = sigma[band_mask]
            v = self._eval_and_fill_single_band(t=t_band, m=m_band, sigma=sigma_band, fill_value=fill_value)
            values.append(np.atleast_1d(v))

        return np.concatenate(values)
