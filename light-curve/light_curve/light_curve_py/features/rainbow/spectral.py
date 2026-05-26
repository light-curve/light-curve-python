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

        # lambda_scale is expressed in Angstrom
        lambda_scale_cm = lambda_scale * 1e-8

        # Phenomenological value for the slope of the extinction
        # No physical reasons for this value, just nice fits
        # Could be improved in the future
        # Fitting instead of fixing is likely overkill for broad band photometry
        intensity = 10**2

        tau = intensity * np.exp(-wave_cm / lambda_scale_cm)

        return base * np.exp(-tau)

    @staticmethod
    def initial_guesses(t, m, sigma, band):
        return {
            "lambda_scale": 10.0,
        }

    @staticmethod
    def limits(t, m, sigma, band):
        return {
            "lambda_scale": (10.0, 1000.0),
        }


spectral_terms = {
    "planck": PlanckSpectralTerm,
    "blanketed": BlanketedPlanckSpectralTerm,
}


