from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Union

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
    def value(t, temp):
        return np.full_like(t, temp)

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
    def value(t, t0, temp_min, temp_max, k_sig):
        dt = t - t0

        # To avoid numerical overflows, let's only compute the exponent not too far from t0
        idx1 = dt <= -100 * k_sig
        idx2 = (dt > -100 * k_sig) & (dt < 100 * k_sig)
        idx3 = dt >= 100 * k_sig

        result = np.zeros(len(dt))
        result[idx1] = temp_max
        result[idx2] = temp_min + (temp_max - temp_min) / (1.0 + np.exp(dt[idx2] / k_sig))
        result[idx3] = temp_min

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


@dataclass
class DelayedSigmoidTemperatureTerm(BaseTemperatureTerm):
    """Sigmoid temperature with delay w.r.t. bolometric peak"""

    @staticmethod
    def parameter_names():
        return ["reference_time", "Tmin", "Tmax", "k_sig", "t_delay"]

    @staticmethod
    def parameter_scalings():
        return ["time", None, None, "timescale", "timescale"]

    @staticmethod
    def value(t, t0, Tmin, Tmax, k_sig, t_delay):
        dt = t - t0 - t_delay

        # To avoid numerical overflows, let's only compute the exponent not too far from t0
        idx1 = dt <= -100 * k_sig
        idx2 = (dt > -100 * k_sig) & (dt < 100 * k_sig)
        idx3 = dt >= 100 * k_sig

        result = np.zeros(len(dt))
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
        initial["t_delay"] = 0.0

        return initial

    @staticmethod
    def limits(t, m, sigma, band):
        t_amplitude = np.ptp(t)

        limits = {}
        limits["Tmin"] = (1e3, 2e6)  # K
        limits["Tmax"] = (1e3, 2e6)  # K
        limits["k_sig"] = (1e-4, 10 * t_amplitude)
        limits["t_delay"] = (-t_amplitude, t_amplitude)

        return limits


temperature_terms = {
    "constant": ConstantTemperatureTerm,
    "sigmoid": SigmoidTemperatureTerm,
    "delayed_sigmoid": DelayedSigmoidTemperatureTerm,
}
