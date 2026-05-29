from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np

__all__ = [
    "spectral_terms",
    "BaseSpectralTerm",
    "PlanckSpectralTerm",
    "BlanketedPlanckSpectralTerm",
]

# CODATA 2018
planck_constant = 6.62607004e-27  # erg s
speed_of_light = 2.99792458e10  # cm/s
boltzman_constant = 1.380649e-16  # erg/K


@dataclass()
class BaseSpectralTerm:
    """Spectral term for Rainbow"""

    @staticmethod
    @abstractmethod
    def parameter_names() -> List[str]:
        return NotImplementedError

    @staticmethod
    @abstractmethod
    def parameter_scalings() -> List[Union[str, None]]:
        return NotImplementedError

    @staticmethod
    @abstractmethod
    def value(wave_cm, T, *params):
        return NotImplementedError

    @staticmethod
    @abstractmethod
    def initial_guesses(t, m, sigma, band) -> Dict[str, float]:
        return NotImplementedError

    @staticmethod
    @abstractmethod
    def limits(t, m, sigma, band) -> Dict[str, float]:
        return NotImplementedError


@dataclass()
class PlanckSpectralTerm(BaseSpectralTerm):
    """Standard blackbody spectrum"""

    @staticmethod
    def parameter_names():
        return []

    @staticmethod
    def parameter_scalings():
        return []

    @staticmethod
    def value(wave_cm, T, *params):
        nu = speed_of_light / wave_cm

        return (
            (2 * planck_constant / speed_of_light**2) * nu**3 / np.expm1(planck_constant * nu / (boltzman_constant * T))
        )

    @staticmethod
    def initial_guesses(t, m, sigma, band):
        return {}

    @staticmethod
    def limits(t, m, sigma, band):
        return {}


@dataclass()
class BlanketedPlanckSpectralTerm(BaseSpectralTerm):
    """Blackbody spectrum with exponential blanketing"""

    @staticmethod
    def parameter_names():
        return ["lambda_scale"]

    @staticmethod
    def parameter_scalings():
        return [None]

    @staticmethod
    def value(wave_cm, T, lambda_scale):
        base = PlanckSpectralTerm.value(wave_cm, T)

        # Phenomenological value for the slope of the extinction
        # No physical reasons for this value, just nice fits
        # Fitting instead of fixing is likely overkill for broad band photometry
        intensity = 10**2

        # This empirical constant encodes how far (in wavelength) the maximum UV extinction extends (lambda_scale=1).
        # The scale of this value comes from how it relates temperature to wavelength in angstrom.
        # This value allows the extinction to affect the BB wavelength a little over the peak (in the formula below).
        max_extinction = 5e7

        # Lambda_angstrom represents how far (in absolute wavelength) the extinction affects the BB.
        # Lambda_scale quantifies this between 0 (no UV ext) and 1 max suppression (encoded by max_extinction above)
        # We want this maximum extinction to scale with the size of the BB.
        # Therefore, lambda_angstrom should inversely scale with temperature.
        lambda_angstrom = max_extinction * lambda_scale / T

        # Convert to cm to match wave_cm
        lambda_cm = lambda_angstrom * 1e-8

        tau = intensity * np.exp(-wave_cm / lambda_cm)

        return base * np.exp(-tau)

    @staticmethod
    def initial_guesses(t, m, sigma, band):
        return {
            "lambda_scale": 0.001,
        }

    @staticmethod
    def limits(t, m, sigma, band):
        return {
            "lambda_scale": (0.001, 1.0),
        }


spectral_terms = {
    "planck": PlanckSpectralTerm,
    "blanketed": BlanketedPlanckSpectralTerm,
}
