from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Union\

import numpy as np

__all__ = [
    "spectral_terms",
    "BaseSpectralTerm",
    "PlanckSpectralTerm",
    "BlanketedPlanckConstTemp",
    "BlanketedPlanckSigmoidTemp",
]

# CODATA 2018
planck_constant = 6.62607004e-27  # erg s
speed_of_light = 2.99792458e10  # cm/s
boltzman_constant = 1.380649e-16  # erg/K
b_wien = 28977720 # Angstrom*K


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
class BlanketedPlanckConstTemp(BaseSpectralTerm):
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
        # Fitting instead of fixing is likely overkill for broad band photometry
        intensity = 100
        
        # Encodes how far (in wavelength) the maximum UV extinction extends (at lambda_scale=1).
        # Allows the extinction to affect the BB wavelength past the peak (in the formula below).
        max_extinction = 2 * b_wien

        # Lambda_angstrom represents how far (in absolute wavelength) the extinction affects the BB.
        # Lambda_scale quantifies this between 0 (no UV ext) and 1 max suppression (encoded by max_extinction above)
        # We want this maximum extinction to scale with the size of the BB.
        # We use T to give the correct order of magntiude of the scaling
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

@dataclass()
class BlanketedPlanckSigmoidTemp(BaseSpectralTerm):
    """Blackbody spectrum with exponential blanketing."""

    @staticmethod
    def parameter_names():
        return ["Tmin", "Tmax", "lambda_scale"]

    @staticmethod
    def parameter_scalings():
        return [None, None, None]

    @staticmethod
    def value(wave_cm, T, Tmin, Tmax, lambda_scale):
        base = PlanckSpectralTerm.value(wave_cm, T)

        # Phenomenological value for the slope of the extinction
        # Fitting instead of fixing is likely overkill for broad band photometry
        intensity = 100
        
        # Encodes how far (in wavelength) the maximum UV extinction extends (at lambda_scale=1).
        # Allows the extinction to affect the BB wavelength past the peak (in the formula below).
        max_extinction = 2 * b_wien

        # The depth of exinction depends on the temperature of source.
        # But it should not vary as the object cools down. We use T(t=~peak)
        T_scale = (Tmin + Tmax) / 2

        # Lambda_angstrom represents how far (in absolute wavelength) the extinction affects the BB.
        # Lambda_scale quantifies this between 0 (no UV ext) and 1 max suppression (encoded by max_extinction above)
        lambda_angstrom = max_extinction * lambda_scale / T_scale

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
    "blanketed_constant_temperature": BlanketedPlanckConstTemp,
    "blanketed_sigmoid_temperature": BlanketedPlanckSigmoidTemp
}
