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
        return ["intensity", "l0"]

    @staticmethod
    def parameter_scalings():
        return [None, None]

    @staticmethod
    def value(wave_cm, T, intensity, l0):
        base = PlanckSpectralTerm.value(wave_cm, T)

        # l0 is expressed in Angstrom
        l0_cm = l0 * 1e-8

        tau = 10**intensity * np.exp(-wave_cm / l0_cm)

        return base * np.exp(-tau)

    @staticmethod
    def initial_guesses(t, m, sigma, band):
        return {
            "intensity": 1.0,
            "l0": 100.0,
        }

    @staticmethod
    def limits(t, m, sigma, band):
        # l0 and intensity are degenerate. Putting either to ~0 leads to a classical BB.
        # We prevent intensity to go to 0, because the blanketing is more physically realistic
        # at low l0 high intensity rather than high l0 low intensity.
        return {
            "intensity": (1.5, 6.0),
            "l0": (100.0, 2000.0),
        }


spectral_terms = {
    "planck": PlanckSpectralTerm,
    "blanketed": BlanketedPlanckSpectralTerm,
}
