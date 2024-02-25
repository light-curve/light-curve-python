from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

__all__ = ["temperature_terms", "BaseTemperatureTerm", "ConstantTemperatureTerm", "SigmoidTemperatureTerm"]


@dataclass()
class BaseTemperatureTerm:
    """Bolometric term for the Rainbow"""

    @staticmethod
    @abstractmethod
    def parameter_names() -> List[str]:
        return NotImplementedError

    @staticmethod
    @abstractmethod
    def parameter_scalings() -> List[str]:
        """Describes how to unscale the parameters - like time, timescale, flux or do not scale"""
        return NotImplementedError

    @staticmethod
    @abstractmethod
    def value(self, t, params) -> float:
        return NotImplementedError

    @staticmethod
    @abstractmethod
    def initial_guesses(t, m, sigma, band) -> Dict[str, float]:
        return NotImplementedError

    @staticmethod
    @abstractmethod
    def limits(t, m, sigma, band) -> Dict[str, float]:
        return NotImplementedError


@dataclass
class ConstantTemperatureTerm(BaseTemperatureTerm):
    """Constant temperature"""

    @staticmethod
    def parameter_names():
        return ["T"]

    @staticmethod
    def parameter_scalings():
        return [None]

    @staticmethod
    def value(t, T):
        return T * np.ones_like(t)

    @staticmethod
    def initial_guesses(t, m, sigma, band):
        initial = {}
        initial["T"] = 8000.0

        return initial

    @staticmethod
    def limits(t, m, sigma, band):
        limits = {}
        limits["T"] = (1e2, 2e6)  # K

        return limits


@dataclass
class SigmoidTemperatureTerm(BaseTemperatureTerm):
    """Sigmoid temperature"""

    @staticmethod
    def parameter_names():
        return ["reference_time", "Tmin", "Tmax", "k_sig"]

    @staticmethod
    def parameter_scalings():
        return ["time", None, None, "timescale"]

    @staticmethod
    def value(t, t0, Tmin, Tmax, k_sig):
        dt = t - t0
        result = np.zeros_like(dt)

        # To avoid numerical overflows, let's only compute the exponent not too far from t0
        idx1 = dt <= -100 * k_sig
        idx2 = (dt > -100 * k_sig) & (dt < 100 * k_sig)
        idx3 = dt >= 100 * k_sig

        result[idx1] = Tmax
        result[idx2] = Tmin + (Tmax - Tmin) / (1.0 + np.exp(dt[idx2] / k_sig))
        result[idx3] = Tmin

        return result

    @staticmethod
    def initial_guesses(t, m, sigma, band):
        initial = {}
        initial["Tmin"] = 7000.0
        initial["Tmax"] = 10000.0
        initial["k_sig"] = 1.0

        return initial

    @staticmethod
    def limits(t, m, sigma, band):
        t_amplitude = np.ptp(t)

        limits = {}
        limits["Tmin"] = (1e3, 2e6)  # K
        limits["Tmax"] = (1e3, 2e6)  # K
        limits["k_sig"] = (1e-4, 10 * t_amplitude)

        return limits


temperature_terms = {
    "constant": ConstantTemperatureTerm,
    "sigmoid": SigmoidTemperatureTerm,
}
