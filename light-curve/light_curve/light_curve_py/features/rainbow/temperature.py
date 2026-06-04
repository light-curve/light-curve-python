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
    """Sigmoid temperature.

    Reparameterized as a peak temperature ``T`` (the early / hot plateau, = old ``Tmax``)
    times a ratio ``T_ratio = Tmin / Tmax`` (the late / cool floor relative to the peak):

    .. math::
        T(t) = T\\,\\big(T_\\mathrm{ratio} + (1 - T_\\mathrm{ratio})\\,s\\big),
        \\quad s = \\frac{1}{1 + e^{(t - t_0) / t_\\mathrm{color}}}

    ``T_ratio = 1`` is a constant temperature ``T``. The independent ``(Tmin, Tmax)`` pair
    is degenerate when the temperature is not actually changing; here ``T`` stays pinned by
    the well-sampled epochs while only the floppy ``T_ratio`` floats, and a weak
    ``N(1, 0.5)`` prior anchors it to 1 (constant) unless the data demand cooling.
    """

    @staticmethod
    def parameter_names():
        return ["reference_time", "T", "T_ratio", "t_color"]

    @staticmethod
    def parameter_scalings():
        return ["time", None, None, "timescale"]

    @staticmethod
    def value(t, t0, T, T_ratio, t_color):
        dt = t - t0

        # To avoid numerical overflows, let's only compute the exponent not too far from t0
        idx1 = dt <= -100 * t_color
        idx2 = (dt > -100 * t_color) & (dt < 100 * t_color)
        idx3 = dt >= 100 * t_color

        result = np.zeros(len(dt))
        result[idx1] = T  # early -> Tmax = T
        result[idx3] = T * T_ratio  # late -> Tmin = T * T_ratio
        s = 1.0 / (1.0 + np.exp(dt[idx2] / t_color))
        result[idx2] = T * (T_ratio + (1.0 - T_ratio) * s)

        return result

    @staticmethod
    def initial_guesses(t, m, sigma, band):
        _, dt = t0_and_weighted_centroid_sigma(t, m, sigma)

        initial = {}
        initial["T"] = 10000.0
        initial["T_ratio"] = 1.0
        initial["t_color"] = 2 * dt

        return initial

    @staticmethod
    def limits(t, m, sigma, band):
        t_amplitude = np.ptp(t)
        _, dt = t0_and_weighted_centroid_sigma(t, m, sigma)

        limits = {}
        limits["T"] = (1e3, 2e6)  # K
        limits["T_ratio"] = (0.1, 5.0)  # Tmin / Tmax
        limits["t_color"] = (dt / 3, 10 * t_amplitude)

        return limits

    @staticmethod
    def parameter_priors():
        # Weak prior anchoring T_ratio to 1 (constant temperature) when unconstrained;
        # in fit space, and T_ratio is unscaled so it equals the physical value.
        return {"T_ratio": (1.0, 0.5)}

    @staticmethod
    def derivatives(t, t0, T, T_ratio, t_color):
        """Jacobian w.r.t. (t0, T, T_ratio, t_color), shape (4, len(t)).

        ``T(t) = T·(T_ratio + (1-T_ratio)·s)``; saturated regions have zero (t0, t_color)
        partials.
        """
        dt = t - t0
        jac = np.zeros((4, len(dt)))

        idx1 = dt <= -100 * t_color  # T(t) == T
        idx2 = (dt > -100 * t_color) & (dt < 100 * t_color)
        idx3 = dt >= 100 * t_color  # T(t) == T * T_ratio

        jac[1, idx1] = 1.0  # ∂T/∂T = 1 (early plateau)
        jac[1, idx3] = T_ratio  # ∂T/∂T (late floor)
        jac[2, idx3] = T  # ∂T/∂T_ratio (late floor)

        if np.any(idx2):
            dt_in = dt[idx2]
            e = np.exp(dt_in / t_color)
            inv_1p_e = 1.0 / (1.0 + e)
            s = inv_1p_e
            s_1ms = e * inv_1p_e * inv_1p_e  # s · (1 - s)
            omr = 1.0 - T_ratio

            jac[0, idx2] = T * omr * s_1ms / t_color  # ∂T/∂t0
            jac[1, idx2] = T_ratio + omr * s  # ∂T/∂T = T(t)/T
            jac[2, idx2] = T * (1.0 - s)  # ∂T/∂T_ratio
            jac[3, idx2] = T * omr * s_1ms * dt_in / (t_color * t_color)  # ∂T/∂t_color

        return jac


@dataclass
class DelayedSigmoidTemperatureTerm(BaseTemperatureTerm):
    """Sigmoid temperature with delay w.r.t. bolometric peak.

    Reparameterized as a peak temperature ``T`` (the early / hot plateau, = old
    ``Tmax``) times a ratio ``T_ratio = Tmin / Tmax`` (the late / cool floor relative to
    the peak), instead of two independent ``(Tmin, Tmax)``:

    .. math::
        T(t) = T\\,\\big(T_\\mathrm{ratio} + (1 - T_\\mathrm{ratio})\\,s\\big),
        \\quad s = \\frac{1}{1 + e^{(t - t_0 - t_\\mathrm{delay}) / t_\\mathrm{color}}}

    ``T_ratio = 1`` is a constant temperature ``T``. When the temperature is not
    actually changing (e.g. TDEs), the absolute ``(Tmin, Tmax)`` pair is degenerate and
    wanders, but here ``T`` stays pinned by the well-sampled epochs while only the
    poorly-constrained ``T_ratio`` floats; a weak ``N(1, 0.5)`` prior then anchors it to 1
    (constant), while genuinely cooling sources, which constrain the ratio, override it.
    The ``t_delay`` carries a weak prior toward 0 in scaled (light-curve-timescale) units
    for the same reason.
    """

    @staticmethod
    def parameter_names():
        return ["reference_time", "T", "T_ratio", "t_color", "t_delay"]

    @staticmethod
    def parameter_scalings():
        return ["time", None, None, "timescale", "timescale"]

    @staticmethod
    def value(t, t0, T, T_ratio, t_color, t_delay):
        dt = t - t0 - t_delay

        # To avoid numerical overflows, let's only compute the exponent not too far from t0
        idx1 = dt <= -100 * t_color
        idx2 = (dt > -100 * t_color) & (dt < 100 * t_color)
        idx3 = dt >= 100 * t_color

        result = np.zeros(len(dt))
        result[idx1] = T  # early -> Tmax = T
        result[idx3] = T * T_ratio  # late -> Tmin = T * T_ratio
        s = 1.0 / (1.0 + np.exp(dt[idx2] / t_color))
        result[idx2] = T * (T_ratio + (1.0 - T_ratio) * s)

        return result

    @staticmethod
    def initial_guesses(t, m, sigma, band):
        _, dt = t0_and_weighted_centroid_sigma(t, m, sigma)

        initial = {}
        initial["T"] = 10000.0
        initial["T_ratio"] = 1.0
        initial["t_color"] = 2 * dt
        initial["t_delay"] = 0.0

        return initial

    @staticmethod
    def limits(t, m, sigma, band):
        t_amplitude = np.ptp(t)
        _, dt = t0_and_weighted_centroid_sigma(t, m, sigma)

        limits = {}
        limits["T"] = (1e3, 2e6)  # K
        limits["T_ratio"] = (0.1, 5.0)  # Tmin / Tmax
        limits["t_color"] = (dt / 3, 10 * t_amplitude)
        limits["t_delay"] = (-t_amplitude, t_amplitude)

        return limits

    @staticmethod
    def parameter_priors():
        # Priors are in fit (scaled) space; T_ratio is unscaled so it equals physical,
        # while the t_delay prior is in light-curve-timescale units (scale = std of times).
        # Both are weak: they only anchor the parameter when the data leave it free
        # (constant-temperature / no-delay sources), and are overridden by real signal.
        return {"T_ratio": (1.0, 0.5), "t_delay": (0.0, 1.0)}

    @staticmethod
    def derivatives(t, t0, T, T_ratio, t_color, t_delay):
        """Jacobian w.r.t. (t0, T, T_ratio, t_color, t_delay), shape (5, len(t)).

        ``T = T·(T_ratio + (1-T_ratio)·s)``; ∂T/∂t_delay equals ∂T/∂t0.
        """
        dt = t - t0 - t_delay
        jac = np.zeros((5, len(dt)))

        idx1 = dt <= -100 * t_color  # T == T
        idx2 = (dt > -100 * t_color) & (dt < 100 * t_color)
        idx3 = dt >= 100 * t_color  # T == T * T_ratio

        jac[1, idx1] = 1.0  # ∂T/∂T = 1 (early plateau)
        jac[1, idx3] = T_ratio  # ∂T/∂T (late floor)
        jac[2, idx3] = T  # ∂T/∂T_ratio (late floor)

        if np.any(idx2):
            dt_in = dt[idx2]
            e = np.exp(dt_in / t_color)
            inv = 1.0 / (1.0 + e)
            s = inv
            s_1ms = e * inv * inv  # s · (1 - s)
            omr = 1.0 - T_ratio

            dT_dt0 = T * omr * s_1ms / t_color
            jac[0, idx2] = dT_dt0
            jac[1, idx2] = T_ratio + omr * s  # = T / T
            jac[2, idx2] = T * (1.0 - s)
            jac[3, idx2] = T * omr * s_1ms * dt_in / (t_color * t_color)
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
