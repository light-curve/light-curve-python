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
        limits["T"] = (1e3, 2e6)  # K

        return limits

    @staticmethod
    def derivatives(t, temp):
        """∂T/∂T = 1, shape (1, len(t))."""
        return np.ones((1, len(t)))


@dataclass
class SigmoidTemperatureTerm(BaseTemperatureTerm):
    """Sigmoid temperature"""

    @staticmethod
    def parameter_names():
        return ["reference_time", "Tmin", "Tmax", "t_color"]

    @staticmethod
    def parameter_scalings():
        return ["time", None, None, "timescale"]

    @staticmethod
    def value(t, t0, temp_min, temp_max, t_color):
        dt = t - t0

        # To avoid numerical overflows, let's only compute the exponent not too far from t0
        idx1 = dt <= -100 * t_color
        idx2 = (dt > -100 * t_color) & (dt < 100 * t_color)
        idx3 = dt >= 100 * t_color

        result = np.zeros(len(dt))
        result[idx1] = temp_max
        result[idx2] = temp_min + (temp_max - temp_min) / (1.0 + np.exp(dt[idx2] / t_color))
        result[idx3] = temp_min

        return result

    @staticmethod
    def initial_guesses(t, m, sigma, band):
        _, dt = t0_and_weighted_centroid_sigma(t, m, sigma)

        initial = {}
        initial["Tmin"] = 7000.0
        initial["Tmax"] = 10000.0
        initial["t_color"] = 2 * dt

        return initial

    @staticmethod
    def limits(t, m, sigma, band):
        t_amplitude = np.ptp(t)
        _, dt = t0_and_weighted_centroid_sigma(t, m, sigma)

        limits = {}
        limits["Tmin"] = (1e3, 2e6)  # K
        limits["Tmax"] = (1e3, 2e6)  # K
        limits["t_color"] = (dt / 3, 10 * t_amplitude)

        return limits

    @staticmethod
    def derivatives(t, t0, temp_min, temp_max, t_color):
        """Jacobian of `value` w.r.t. (t0, Tmin, Tmax, t_color), shape (4, len(t)).

        Mirrors the three-region clamping in `value`: in the saturated regions
        the value is constant in (t0, t_color), so those partials are zero.
        """
        dt = t - t0
        jac = np.zeros((4, len(dt)))

        idx1 = dt <= -100 * t_color  # T == Tmax
        idx2 = (dt > -100 * t_color) & (dt < 100 * t_color)
        idx3 = dt >= 100 * t_color  # T == Tmin

        jac[2, idx1] = 1.0
        jac[1, idx3] = 1.0

        if np.any(idx2):
            dt_in = dt[idx2]
            e = np.exp(dt_in / t_color)
            inv_1p_e = 1.0 / (1.0 + e)
            s = inv_1p_e  # sigmoid(-dt/t_color)
            s_1ms = e * inv_1p_e * inv_1p_e  # s * (1 - s)
            delta_t = temp_max - temp_min

            # ∂T/∂t0   =  ΔT · s(1-s) / t_color
            jac[0, idx2] = delta_t * s_1ms / t_color
            # ∂T/∂Tmin =  1 - s
            jac[1, idx2] = 1.0 - s
            # ∂T/∂Tmax =  s
            jac[2, idx2] = s
            # ∂T/∂t_color = ΔT · s(1-s) · dt / t_color²
            jac[3, idx2] = delta_t * s_1ms * dt_in / (t_color * t_color)

        return jac


@dataclass
class DelayedSigmoidTemperatureTerm(BaseTemperatureTerm):
    """Sigmoid temperature with delay w.r.t. bolometric peak"""

    @staticmethod
    def parameter_names():
        return ["reference_time", "Tmin", "Tmax", "t_color", "t_delay"]

    @staticmethod
    def parameter_scalings():
        return ["time", None, None, "timescale", "timescale"]

    @staticmethod
    def value(t, t0, Tmin, Tmax, t_color, t_delay):
        dt = t - t0 - t_delay

        # To avoid numerical overflows, let's only compute the exponent not too far from t0
        idx1 = dt <= -100 * t_color
        idx2 = (dt > -100 * t_color) & (dt < 100 * t_color)
        idx3 = dt >= 100 * t_color

        result = np.zeros(len(dt))
        result[idx1] = Tmax
        result[idx2] = Tmin + (Tmax - Tmin) / (1.0 + np.exp(dt[idx2] / t_color))
        result[idx3] = Tmin

        return result

    @staticmethod
    def initial_guesses(t, m, sigma, band):
        _, dt = t0_and_weighted_centroid_sigma(t, m, sigma)

        initial = {}
        initial["Tmin"] = 7000.0
        initial["Tmax"] = 10000.0
        initial["t_color"] = 2 * dt
        initial["t_delay"] = 0.0

        return initial

    @staticmethod
    def limits(t, m, sigma, band):
        t_amplitude = np.ptp(t)
        _, dt = t0_and_weighted_centroid_sigma(t, m, sigma)

        limits = {}
        limits["Tmin"] = (1e3, 2e6)  # K
        limits["Tmax"] = (1e3, 2e6)  # K
        limits["t_color"] = (dt / 3, 10 * t_amplitude)
        limits["t_delay"] = (-t_amplitude, t_amplitude)

        return limits

    @staticmethod
    def derivatives(t, t0, Tmin, Tmax, t_color, t_delay):
        """Jacobian, shape (5, len(t)). Same body as SigmoidTemperatureTerm
        but ``dt = t - t0 - t_delay``, so ∂T/∂t_delay equals ∂T/∂t0.
        """
        dt = t - t0 - t_delay
        jac = np.zeros((5, len(dt)))

        idx1 = dt <= -100 * t_color
        idx2 = (dt > -100 * t_color) & (dt < 100 * t_color)
        idx3 = dt >= 100 * t_color

        jac[2, idx1] = 1.0
        jac[1, idx3] = 1.0

        if np.any(idx2):
            dt_in = dt[idx2]
            e = np.exp(dt_in / t_color)
            inv = 1.0 / (1.0 + e)
            s = inv
            s_1ms = e * inv * inv
            delta_t = Tmax - Tmin

            dT_dt0 = delta_t * s_1ms / t_color
            jac[0, idx2] = dT_dt0
            jac[1, idx2] = 1.0 - s
            jac[2, idx2] = s
            jac[3, idx2] = delta_t * s_1ms * dt_in / (t_color * t_color)
            jac[4, idx2] = dT_dt0  # ∂T/∂t_delay = ∂T/∂t0

        return jac


def median_dt(t, band):
    # Compute the median distance between points in each band
    dt = []
    for b in np.unique(band):
        dt += list(t[band == b][1:] - t[band == b][:-1])
    med_dt = np.median(dt)
    return med_dt


def t0_and_weighted_centroid_sigma(t, m, sigma):
    # To avoid crashing on all-negative data
    mc = m - np.min(m)

    # Peak position as weighted centroid of everything above median
    idx = m > np.median(m)
    t0 = np.sum(t[idx] * m[idx] / sigma[idx]) / np.sum(m[idx] / sigma[idx])

    # Weighted centroid sigma
    dt = np.sqrt(np.sum((t[idx] - t0) ** 2 * (mc[idx]) / sigma[idx]) / np.sum(mc[idx] / sigma[idx]))
    return t0, dt


temperature_terms = {
    "constant": ConstantTemperatureTerm,
    "sigmoid": SigmoidTemperatureTerm,
    "delayed_sigmoid": DelayedSigmoidTemperatureTerm,
}
