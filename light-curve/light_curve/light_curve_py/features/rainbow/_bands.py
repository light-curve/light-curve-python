from typing import Any, Dict

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Bands:
    """Monochromatic passbands for Rainbow fit."""

    @property
    def names(self) -> NDArray:
        """Names of the bands"""
        return self._names.copy()

    @property
    def index(self) -> NDArray[int]:
        """Index array for the bands"""
        return np.arange(len(self._names))

    def wave_cm(self, band) -> float:
        """Wavelength of the band in cm"""
        return self._names_to_wave_cm[band]

    @property
    def mean_wave_cm(self) -> float:
        """Mean wavelength of the bands in cm"""
        return np.mean(self._wave_cm)

    def index_to_wave_cm(self, index: NDArray[int]) -> NDArray[float]:
        """Wavelength of the band in cm"""
        return self._wave_cm[index]

    def __init__(self, names: ArrayLike, wave_cm: ArrayLike):
        self._input_validation(names, wave_cm)

        self._names = np.asarray(names)
        self._wave_cm = np.asarray(wave_cm)

        self._names_to_wave_cm = dict(zip(self._names, self._wave_cm))

        self._name_to_index = dict(zip(self._names, range(len(self._names))))
        self.get_index = np.vectorize(self._name_to_index.get)

    @classmethod
    def from_dict(cls, band_wave_cm: Dict[Any, float]) -> "Bands":
        """Create Bands from a dictionary"""
        names, wave_cm = zip(*band_wave_cm.items())
        return cls(names, wave_cm)

    @staticmethod
    def _input_validation(names: ArrayLike, wave_cm: ArrayLike):
        if len(names) != len(wave_cm):
            raise ValueError("names and wave_cm must have the same length")

        if len(names) == 0:
            raise ValueError("At least one band must be specified.")

        if len(set(names)) != len(names):
            raise ValueError("names must be unique")

        if any(lmbd <= 0 for lmbd in wave_cm):
            raise ValueError("wave_cm must be positive")
