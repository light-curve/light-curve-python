from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Union

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
    def parameter_scalings() -> List[Union[str, None]]:
        """Describes how to unscale the parameters.

        Should be the list the same shape as returned by `parameter_names()`, and describes
        how the parameters should be un-scaled from the fit done in scaled coordinates.

        List items should be either None or one of the following strings:
        - time - the parameter is scaled and shifted like measurement times
        - timescale - the parameter is scaled like measurement times, but not shifted, thus
            behaving like a difference between two measurement times
        - flux - the parameter is scaled like the flux points, without additional shifts
            applied to them. Suitable for amplitude-like parameters.
        - None - the parameter is kept as is, without any additional scaling or shifting
        """
        return NotImplementedError

    @staticmethod
    @abstractmethod
    def value(self, t, *params) -> float:
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
    def peak_time(*params) -> float:
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

        result = np.zeros(len(dt))
        # To avoid numerical overflows, let's only compute the exponents not too far from t0
        idx = dt > -100 * rise_time
        result[idx] = amplitude / (np.exp(-dt[idx] / rise_time) + 1)

        return result

    @staticmethod
    def initial_guesses(t, m, sigma, band):
        A = np.ptp(m)

        initial = {}
        initial["reference_time"] = t[np.argmax(m)]
        initial["amplitude"] = A
        initial["rise_time"] = 1.0

        return initial

    @staticmethod
    def limits(t, m, sigma, band):
        t_amplitude = np.ptp(t)
        m_amplitude = np.ptp(m)

        mean_dt = np.median(t[1:] - t[:-1])

        limits = {}
        limits["reference_time"] = (np.min(t) - 10 * t_amplitude, np.max(t) + 10 * t_amplitude)
        limits["amplitude"] = (0.0, 20 * m_amplitude)
        limits["rise_time"] = (0.1 * mean_dt, 10 * t_amplitude)

        return limits

    @staticmethod
    def peak_time(t0, amplitude, rise_time):
        """Peak time is not defined for the sigmoid, so it returns mid-time of the rise instead"""
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

        result = np.zeros(len(dt))
        # To avoid numerical overflows, let's only compute the exponents not too far from t0
        idx = (dt > -100 * rise_time) & (dt < 100 * fall_time)
        result[idx] = amplitude * scale / (np.exp(-dt[idx] / rise_time) + np.exp(dt[idx] / fall_time))

        return result

    @staticmethod
    def initial_guesses(t, m, sigma, band):
        A = np.ptp(m)

        mc = m - np.min(m)  # To avoid crashing on all-negative data

        # Naive peak position from the highest point
        t0 = t[np.argmax(m)]
        # Peak position as weighted centroid of everything above median
        idx = m > np.median(m)
        # t0 = np.sum(t[idx] * m[idx] / sigma[idx]) / np.sum(m[idx] / sigma[idx])
        # Weighted centroid sigma
        dt = np.sqrt(np.sum((t[idx] - t0) ** 2 * (mc[idx]) / sigma[idx]) / np.sum(mc[idx] / sigma[idx]))

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
        m_amplitude = np.ptp(m)

        mean_dt = np.median(t[1:] - t[:-1])

        limits = {}
        limits["reference_time"] = (np.min(t) - 10 * t_amplitude, np.max(t) + 10 * t_amplitude)
        limits["amplitude"] = (0.0, 20 * m_amplitude)
        limits["rise_time"] = (0.1 * mean_dt, 10 * t_amplitude)
        limits["fall_time"] = (0.1 * mean_dt, 10 * t_amplitude)

        return limits

    @staticmethod
    def peak_time(t0, amplitude, rise_time, fall_time):
        return t0 + np.log(fall_time / rise_time) * rise_time * fall_time / (rise_time + fall_time)


bolometric_terms = {
    "sigmoid": SigmoidBolometricTerm,
    "bazin": BazinBolometricTerm,
}
