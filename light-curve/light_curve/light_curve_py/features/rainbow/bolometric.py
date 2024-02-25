from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

__all__ = ["bolometric_terms", "BaseBolometricTerm", "SigmoidBolometricTerm", "BazinBolometricTerm"]


@dataclass()
class BaseBolometricTerm:
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

    @staticmethod
    @abstractmethod
    def peak_time(params) -> float:
        return NotImplementedError


@dataclass()
class SigmoidBolometricTerm(BaseBolometricTerm):
    """Sigmoid"""

    @staticmethod
    def parameter_names():
        return ["reference_time", "amplitude", "rise_time"]

    @staticmethod
    def parameter_scalings():
        return ["time", "flux", "timescale"]

    @staticmethod
    def value(t, t0, amplitude, rise_time):
        dt = t - t0

        result = np.zeros_like(dt)
        # To avoid numerical overflows, let's only compute the exponents not too far from t0
        idx = dt > -100 * rise_time
        result[idx] = amplitude / (np.exp(-dt[idx] / rise_time) + 1)

        return result

    @staticmethod
    def initial_guesses(t, m, sigma, band):
        A = np.max(m)

        initial = {}
        initial["reference_time"] = t[np.argmax(m)]
        initial["amplitude"] = A
        initial["rise_time"] = 1.0

        return initial

    @staticmethod
    def limits(t, m, sigma, band):
        t_amplitude = np.ptp(t)
        m_amplitude = np.max(m)

        limits = {}
        limits["reference_time"] = (np.min(t) - 10 * t_amplitude, np.max(t) + 10 * t_amplitude)
        limits["amplitude"] = (0.0, 10 * m_amplitude)
        limits["rise_time"] = (1e-4, 10 * t_amplitude)

        return limits

    @staticmethod
    def peak_time(t0, amplitude, rise_time):
        return t0


@dataclass()
class BazinBolometricTerm(BaseBolometricTerm):
    """Bazin function, symmetric form"""

    @staticmethod
    def parameter_names():
        return ["reference_time", "amplitude", "rise_time", "fall_time"]

    @staticmethod
    def parameter_scalings():
        return ["time", "flux", "timescale", "timescale"]

    @staticmethod
    def value(t, t0, amplitude, rise_time, fall_time):
        dt = t - t0

        # Coefficient to make peak amplitude equal to unity
        scale = (fall_time / rise_time) ** (rise_time / (fall_time + rise_time)) + (fall_time / rise_time) ** (
            -fall_time / (fall_time + rise_time)
        )

        result = np.zeros_like(dt)
        # To avoid numerical overflows, let's only compute the exponents not too far from t0
        idx = (dt > -100 * rise_time) & (dt < 100 * fall_time)
        result[idx] = amplitude * scale / (np.exp(-dt[idx] / rise_time) + np.exp(dt[idx] / fall_time))

        return result

    @staticmethod
    def initial_guesses(t, m, sigma, band):
        A = np.max(m)

        # Naive peak position from the highest point
        t0 = t[np.argmax(m)]
        # Peak position as weighted centroid of everything above zero
        idx = m > 0
        # t0 = np.sum(t[idx] * m[idx] / sigma[idx]) / np.sum(m[idx] / sigma[idx])
        # Weighted centroid sigma
        dt = np.sqrt(np.sum((t[idx] - t0) ** 2 * m[idx] / sigma[idx]) / np.sum(m[idx] / sigma[idx]))

        # Empirical conversion of sigma to rise/fall times
        rise_time = dt / 2
        fall_time = dt / 2

        # Compensate for the difference between reference_time and peak position
        t0 -= np.log(fall_time / rise_time) * rise_time * fall_time / (rise_time + fall_time)

        initial = {}
        initial["reference_time"] = t0
        initial["amplitude"] = A
        initial["rise_time"] = rise_time
        initial["fall_time"] = fall_time

        return initial

    @staticmethod
    def limits(t, m, sigma, band):
        t_amplitude = np.ptp(t)
        m_amplitude = np.max(m)

        limits = {}
        limits["reference_time"] = (np.min(t) - 10 * t_amplitude, np.max(t) + 10 * t_amplitude)
        limits["amplitude"] = (0.0, 10 * m_amplitude)
        limits["rise_time"] = (1e-4, 10 * t_amplitude)
        limits["fall_time"] = (1e-4, 10 * t_amplitude)

        return limits

    @staticmethod
    def peak_time(t0, amplitude, rise_time, fall_time):
        return t0 + np.log(fall_time / rise_time) * rise_time * fall_time / (rise_time + fall_time)


bolometric_terms = {
    "sigmoid": SigmoidBolometricTerm,
    "bazin": BazinBolometricTerm,
}
