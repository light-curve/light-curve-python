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

    Parametrized by the mid temperature ``T = (Tmin + Tmax) / 2`` and a dimensionless
    relative amplitude ``T_amplitude = (Tmax - Tmin) / (Tmax + Tmin)``, so that
    ``Tmax = T(1 + T_amplitude)`` and ``Tmin = T(1 - T_amplitude)``:

    .. math::
        T(t) = T\\,\\big(1 + T_\\mathrm{amplitude}\\,(2 s - 1)\\big),
        \\quad s = \\frac{1}{1 + e^{(t - t_0) / t_\\mathrm{color}}}

    ``s`` runs from 1 (early, hot) to 0 (late, cool), so ``T(t)`` runs from ``Tmax`` to
    ``Tmin``. ``T_amplitude = 0`` is a constant temperature ``T``; a weak
    ``N(0, _t_amplitude_prior_sigma)`` prior anchors it to 0 unless the data demand a
    temperature change. Strictly positive temperature corresponds to the independent box
    ``T > 0``, ``-1 < T_amplitude < 1``.
    """

    # Width of the Gaussian prior anchoring T_amplitude to 0 (constant temperature). Smaller
    # => the constant-temperature classes (TDE, AGN) tie more tightly to 0, at the cost of
    # some sensitivity to genuine cooling; the cooling signal is robust down to ~0.25.
    _t_amplitude_prior_sigma = 0.25

    @staticmethod
    def parameter_names():
        return ["reference_time", "T", "T_amplitude", "t_color"]

    @staticmethod
    def parameter_scalings():
        return ["time", None, None, "timescale"]

    @staticmethod
    def value(t, t0, T, T_amplitude, t_color):
        dt = t - t0

        # To avoid numerical overflows, let's only compute the exponent not too far from t0
        idx1 = dt <= -100 * t_color
        idx2 = (dt > -100 * t_color) & (dt < 100 * t_color)
        idx3 = dt >= 100 * t_color

        result = np.zeros(len(dt))
        result[idx1] = T * (1.0 + T_amplitude)  # early -> Tmax
        result[idx3] = T * (1.0 - T_amplitude)  # late -> Tmin
        s = 1.0 / (1.0 + np.exp(dt[idx2] / t_color))
        result[idx2] = T * (1.0 + T_amplitude * (2.0 * s - 1.0))

        return result

    @staticmethod
    def initial_guesses(t, m, sigma, band):
        _, dt = t0_and_weighted_centroid_sigma(t, m, sigma)

        initial = {}
        initial["T"] = 10000.0
        initial["T_amplitude"] = 0.0
        initial["t_color"] = 2 * dt

        return initial

    @staticmethod
    def limits(t, m, sigma, band):
        t_amplitude = np.ptp(t)
        _, dt = t0_and_weighted_centroid_sigma(t, m, sigma)

        limits = {}
        limits["T"] = (1e3, 2e6)  # K (mid temperature)
        limits["T_amplitude"] = (-0.99, 0.99)  # (Tmax - Tmin) / (Tmax + Tmin); |.| < 1 keeps T(t) > 0
        limits["t_color"] = (dt / 3, 10 * t_amplitude)

        return limits

    @staticmethod
    def parameter_priors():
        # Weak prior anchoring T_amplitude to 0 (constant temperature) when unconstrained;
        # in fit space T_amplitude is unscaled so it equals the physical value.
        return {"T_amplitude": (0.0, SigmoidTemperatureTerm._t_amplitude_prior_sigma)}

    @staticmethod
    def derivatives(t, t0, T, T_amplitude, t_color):
        """Jacobian w.r.t. (t0, T, T_amplitude, t_color), shape (4, len(t)).

        ``T(t) = T·(1 + T_amplitude·(2s - 1))``; saturated regions have zero (t0, t_color)
        partials.
        """
        dt = t - t0
        jac = np.zeros((4, len(dt)))

        idx1 = dt <= -100 * t_color  # T(t) == T·(1 + T_amplitude)
        idx2 = (dt > -100 * t_color) & (dt < 100 * t_color)
        idx3 = dt >= 100 * t_color  # T(t) == T·(1 - T_amplitude)

        jac[1, idx1] = 1.0 + T_amplitude  # ∂T/∂T (early plateau)
        jac[2, idx1] = T  # ∂T/∂T_amplitude (early plateau)
        jac[1, idx3] = 1.0 - T_amplitude  # ∂T/∂T (late floor)
        jac[2, idx3] = -T  # ∂T/∂T_amplitude (late floor)

        if np.any(idx2):
            dt_in = dt[idx2]
            e = np.exp(dt_in / t_color)
            inv_1p_e = 1.0 / (1.0 + e)
            s = inv_1p_e
            s_1ms = e * inv_1p_e * inv_1p_e  # s · (1 - s)
            two_s_m1 = 2.0 * s - 1.0

            jac[0, idx2] = 2.0 * T * T_amplitude * s_1ms / t_color  # ∂T/∂t0
            jac[1, idx2] = 1.0 + T_amplitude * two_s_m1  # ∂T/∂T = T(t)/T
            jac[2, idx2] = T * two_s_m1  # ∂T/∂T_amplitude
            jac[3, idx2] = 2.0 * T * T_amplitude * s_1ms * dt_in / (t_color * t_color)  # ∂T/∂t_color

        return jac


@dataclass
class DelayedSigmoidTemperatureTerm(BaseTemperatureTerm):
    """Sigmoid temperature with delay w.r.t. bolometric peak.

    Parametrized by the mid temperature ``T = (Tmin + Tmax) / 2`` and a dimensionless
    relative amplitude ``T_amplitude = (Tmax - Tmin) / (Tmax + Tmin)``, so that
    ``Tmax = T(1 + T_amplitude)`` and ``Tmin = T(1 - T_amplitude)``:

    .. math::
        T(t) = T\\,\\big(1 + T_\\mathrm{amplitude}\\,(2 s - 1)\\big),
        \\quad s = \\frac{1}{1 + e^{(t - t_0 - t_\\mathrm{delay}) / t_\\mathrm{color}}}

    ``T_amplitude = 0`` is a constant temperature ``T``. When the temperature is not
    actually changing (e.g. TDEs) the swing is degenerate and wanders, so a weak
    ``N(0, _t_amplitude_prior_sigma)`` prior anchors ``T_amplitude`` to 0 while ``T`` stays
    pinned by the well-sampled epochs; genuinely cooling sources, which constrain the
    amplitude, override it. The ``t_delay`` carries a weak prior toward 0 in scaled
    (light-curve-timescale) units for the same reason. Strictly positive temperature
    corresponds to the independent box ``T > 0``, ``-1 < T_amplitude < 1``.
    """

    # Width of the Gaussian prior anchoring T_amplitude to 0 (constant temperature); see
    # SigmoidTemperatureTerm for the anchoring-vs-cooling-sensitivity trade-off.
    _t_amplitude_prior_sigma = 0.25

    @staticmethod
    def parameter_names():
        return ["reference_time", "T", "T_amplitude", "t_color", "t_delay"]

    @staticmethod
    def parameter_scalings():
        return ["time", None, None, "timescale", "timescale"]

    @staticmethod
    def value(t, t0, T, T_amplitude, t_color, t_delay):
        dt = t - t0 - t_delay

        # To avoid numerical overflows, let's only compute the exponent not too far from t0
        idx1 = dt <= -100 * t_color
        idx2 = (dt > -100 * t_color) & (dt < 100 * t_color)
        idx3 = dt >= 100 * t_color

        result = np.zeros(len(dt))
        result[idx1] = T * (1.0 + T_amplitude)  # early -> Tmax
        result[idx3] = T * (1.0 - T_amplitude)  # late -> Tmin
        s = 1.0 / (1.0 + np.exp(dt[idx2] / t_color))
        result[idx2] = T * (1.0 + T_amplitude * (2.0 * s - 1.0))

        return result

    @staticmethod
    def initial_guesses(t, m, sigma, band):
        _, dt = t0_and_weighted_centroid_sigma(t, m, sigma)

        initial = {}
        initial["T"] = 10000.0
        initial["T_amplitude"] = 0.0
        initial["t_color"] = 2 * dt
        initial["t_delay"] = 0.0

        return initial

    @staticmethod
    def limits(t, m, sigma, band):
        t_amplitude = np.ptp(t)
        _, dt = t0_and_weighted_centroid_sigma(t, m, sigma)

        limits = {}
        limits["T"] = (1e3, 2e6)  # K (mid temperature)
        limits["T_amplitude"] = (-0.99, 0.99)  # (Tmax - Tmin) / (Tmax + Tmin); |.| < 1 keeps T(t) > 0
        limits["t_color"] = (dt / 3, 10 * t_amplitude)
        limits["t_delay"] = (-t_amplitude, t_amplitude)

        return limits

    @staticmethod
    def parameter_priors():
        # Priors are in fit (scaled) space; T_amplitude is unscaled so it equals physical,
        # while the t_delay prior is in light-curve-timescale units (scale = std of times).
        # Both are weak: they only anchor the parameter when the data leave it free
        # (constant-temperature / no-delay sources), and are overridden by real signal.
        return {"T_amplitude": (0.0, DelayedSigmoidTemperatureTerm._t_amplitude_prior_sigma), "t_delay": (0.0, 1.0)}

    @staticmethod
    def derivatives(t, t0, T, T_amplitude, t_color, t_delay):
        """Jacobian w.r.t. (t0, T, T_amplitude, t_color, t_delay), shape (5, len(t)).

        ``T(t) = T·(1 + T_amplitude·(2s - 1))``; ∂T/∂t_delay equals ∂T/∂t0.
        """
        dt = t - t0 - t_delay
        jac = np.zeros((5, len(dt)))

        idx1 = dt <= -100 * t_color  # T(t) == T·(1 + T_amplitude)
        idx2 = (dt > -100 * t_color) & (dt < 100 * t_color)
        idx3 = dt >= 100 * t_color  # T(t) == T·(1 - T_amplitude)

        jac[1, idx1] = 1.0 + T_amplitude  # ∂T/∂T (early plateau)
        jac[2, idx1] = T  # ∂T/∂T_amplitude (early plateau)
        jac[1, idx3] = 1.0 - T_amplitude  # ∂T/∂T (late floor)
        jac[2, idx3] = -T  # ∂T/∂T_amplitude (late floor)

        if np.any(idx2):
            dt_in = dt[idx2]
            e = np.exp(dt_in / t_color)
            inv = 1.0 / (1.0 + e)
            s = inv
            s_1ms = e * inv * inv  # s · (1 - s)
            two_s_m1 = 2.0 * s - 1.0

            dT_dt0 = 2.0 * T * T_amplitude * s_1ms / t_color
            jac[0, idx2] = dT_dt0
            jac[1, idx2] = 1.0 + T_amplitude * two_s_m1  # = T(t) / T
            jac[2, idx2] = T * two_s_m1
            jac[3, idx2] = 2.0 * T * T_amplitude * s_1ms * dt_in / (t_color * t_color)
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
