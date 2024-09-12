from dataclasses import dataclass
from typing import Dict, Union

import numpy as np
from numpy.typing import NDArray

__all__ = ["Scaler", "MultiBandScaler"]


@dataclass()
class Scaler:
    """Shift and scale arrays"""

    shift: Union[float, NDArray]
    """Scale to apply to the array

    Either a single value or an array of the same shape as the input array
    """

    scale: Union[float, NDArray]
    """Scale to apply to the array

    Either a single value or an array of the same shape as the input array
    """

    @classmethod
    def from_time(cls, t) -> "Scaler":
        """Create a Scaler from a time array

        It just computes the mean and standard deviation of the array.
        """
        shift = np.mean(t)
        scale = np.std(t)
        if scale == 0.0:
            scale = 1.0
        return cls(shift=shift, scale=scale)

    def do_shift_scale(self, x):
        return (x - self.shift) / self.scale

    def undo_shift_scale(self, x):
        return x * self.scale + self.shift

    def do_scale(self, x):
        return x / self.scale

    def undo_scale(self, x):
        return x * self.scale


@dataclass()
class MultiBandScaler(Scaler):
    """Shift and scale arrays, optionally per band"""

    per_band_shift: Dict[str, float]
    """Shift to apply to each band"""

    @classmethod
    def from_flux(cls, flux, band, *, with_baseline: bool) -> "MultiBandScaler":
        """Create a Scaler from a flux array.

        It uses standard deviation for the scale. For the shift, it is either
        zero (`with_baseline=False`) or the mean of each band otherwise.
        """
        uniq_bands = np.unique(band)
        per_band_shift = dict.fromkeys(uniq_bands, 0.0)
        shift_array = np.zeros(len(flux))

        if with_baseline:
            for b in uniq_bands:
                idx = band == b
                shift_array[idx] = per_band_shift[b] = np.mean(flux[idx])

        scale = np.std(flux)
        if scale == 0.0:
            scale = 1.0

        return cls(shift=shift_array, scale=scale, per_band_shift=per_band_shift)

    def undo_shift_scale_band(self, x, band):
        return x * self.scale + self.per_band_shift.get(band, 0)
