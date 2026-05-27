import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np

__all__ = [
    "bolometric_terms",
    "BaseBolometricTerm",
    "SigmoidBolometricTerm",
    "BazinBolometricTerm",
    "LinexpBolometricTerm",
    "DoublexpBolometricTerm",
]


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

        _, dt = t0_and_weighted_centroid_sigma(t, m, sigma)

        limits = {}
        limits["reference_time"] = (np.min(t) - 10 * t_amplitude, np.max(t) + 10 * t_amplitude)
        limits["amplitude"] = (0.0, 20 * m_amplitude)
        limits["rise_time"] = (dt / 100, 10 * t_amplitude)

        return limits

    @staticmethod
    def peak_time(t0, amplitude, rise_time):
        """Peak time is not defined for the sigmoid, so it returns mid-time of the rise instead"""
        return t0

    @staticmethod
    def derivatives(t, t0, amplitude, rise_time):
        """Jacobian of `value` w.r.t. (t0, amplitude, rise_time), shape (3, len(t))."""
        dt = t - t0
        jac = np.zeros((3, len(dt)))

        idx = dt > -100 * rise_time
        if not np.any(idx):
            return jac

        dt_in = dt[idx]
        e = np.exp(-dt_in / rise_time)
        s = 1.0 / (e + 1.0)
        s_1ms = e * s * s  # s · (1 - s)

        # ∂B/∂t0   = -A · s(1-s) / τ
        jac[0, idx] = -amplitude * s_1ms / rise_time
        # ∂B/∂A    = s
        jac[1, idx] = s
        # ∂B/∂τ    = -A · s(1-s) · dt / τ²
        jac[2, idx] = -amplitude * s_1ms * dt_in / (rise_time * rise_time)

        return jac


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
        A = 1.5 * max(np.max(m), np.ptp(m))

        t0, dt = t0_and_weighted_centroid_sigma(t, m, sigma)

        # Empirical conversion of sigma to rise/fall times
        rise_time = dt
        fall_time = dt

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
        _, dt = t0_and_weighted_centroid_sigma(t, m, sigma)

        limits = {}
        limits["reference_time"] = (np.min(t) - 10 * t_amplitude, np.max(t) + 10 * t_amplitude)
        limits["amplitude"] = (0.0, 20 * m_amplitude)
        limits["rise_time"] = (dt / 100, 10 * t_amplitude)
        limits["fall_time"] = (dt / 100, 10 * t_amplitude)

        return limits

    @staticmethod
    def peak_time(t0, amplitude, rise_time, fall_time):
        return t0 + np.log(fall_time / rise_time) * rise_time * fall_time / (rise_time + fall_time)

    @staticmethod
    def derivatives(t, t0, amplitude, rise_time, fall_time):
        """Jacobian of `value` w.r.t. (t0, amplitude, rise_time, fall_time).

        Returns an array of shape (4, len(t)) whose rows match the order of
        `parameter_names()`. Outside the same clamping window used by `value`,
        all partials are zero (the value is identically zero there).
        """
        dt = t - t0
        jac = np.zeros((4, len(dt)))

        idx = (dt > -100 * rise_time) & (dt < 100 * fall_time)
        if not np.any(idx):
            return jac

        dt_in = dt[idx]
        e_r = np.exp(-dt_in / rise_time)
        e_f = np.exp(dt_in / fall_time)
        denom = e_r + e_f

        # Scale factor (peak == amplitude) and its partials in τ_r, τ_f.
        alpha = fall_time / rise_time
        tau = rise_time + fall_time
        u = rise_time / tau
        v = fall_time / tau
        log_alpha = np.log(alpha)
        a1 = alpha**u
        a2 = alpha ** (-v)
        scale = a1 + a2
        # d(α^u)/dτ_r and d(α^-v)/dτ_r — see analysis notes.
        dscale_dr = a1 * (v * log_alpha / tau - u / rise_time) + a2 * (v * log_alpha / tau + v / rise_time)
        dscale_df = a1 * u * (1.0 / fall_time - log_alpha / tau) + a2 * (-u * log_alpha / tau - v / fall_time)

        b = amplitude * scale / denom

        # ∂B/∂t0 = (B/D) * (E_f/τ_f - E_r/τ_r)
        jac[0, idx] = (b / denom) * (e_f / fall_time - e_r / rise_time)
        # ∂B/∂A = scale / D
        jac[1, idx] = scale / denom
        # ∂B/∂τ_r = B * (dscale_dr/scale  -  E_r * dt / (τ_r² · D))
        jac[2, idx] = b * (dscale_dr / scale - e_r * dt_in / (rise_time**2 * denom))
        # ∂B/∂τ_f = B * (dscale_df/scale  +  E_f * dt / (τ_f² · D))
        jac[3, idx] = b * (dscale_df / scale + e_f * dt_in / (fall_time**2 * denom))

        return jac


@dataclass()
class LinexpBolometricTerm(BaseBolometricTerm):
    """Linexp function, symmetric form. Generated using a prototype version of Multi-view
    Symbolic Regression (Russeil et al. 2024, https://arxiv.org/abs/2402.04298) on
    a SLSN ZTF light curve (https://ztf.snad.space/dr17/view/821207100004043). Careful not very stable guesses/limits"""

    @staticmethod
    def parameter_names():
        return ["reference_time", "amplitude", "rise_time"]

    @staticmethod
    def parameter_scalings():
        return ["time", "flux", "timescale"]

    @staticmethod
    def value(t, t0, amplitude, rise_time):
        dt = t0 - t
        protected_rise = math.copysign(max(1e-5, abs(rise_time)), rise_time)

        # Coefficient to make peak amplitude equal to unity
        scale = 1 / (protected_rise * np.exp(-1))

        power = -dt / protected_rise
        power = np.where(power > 100, 100, power)
        result = amplitude * scale * dt * np.exp(power)

        return np.where(result > 0, result, 0)

    @staticmethod
    def initial_guesses(t, m, sigma, band):
        A = np.ptp(m)
        med_dt = median_dt(t, band)
        t0, dt = t0_and_weighted_centroid_sigma(t, m, sigma)

        # Compute points after or before maximum
        peak_time = t[np.argmax(m)]
        after = t[-1] - peak_time
        before = peak_time - t[0]

        rise_time = 100 * med_dt
        rise_time = rise_time if before >= after else -rise_time

        initial = {}
        # Reference of linexp correspond to the moment where flux == 0
        initial["reference_time"] = peak_time + rise_time
        initial["amplitude"] = A
        initial["rise_time"] = rise_time

        return initial

    @staticmethod
    def limits(t, m, sigma, band):
        t_amplitude = np.ptp(t)
        m_amplitude = np.ptp(m)

        limits = {}
        limits["reference_time"] = (np.min(t) - 10 * t_amplitude, np.max(t) + 10 * t_amplitude)
        limits["amplitude"] = (0, 10 * m_amplitude)
        limits["rise_time"] = (-10 * t_amplitude, 10 * t_amplitude)

        return limits

    @staticmethod
    def peak_time(t0, amplitude, rise_time):
        return t0 - rise_time

    @staticmethod
    def derivatives(t, t0, amplitude, rise_time):
        """Jacobian of `value` w.r.t. (t0, amplitude, rise_time), shape (3, len(t)).

        Inactive region (where `value` clips to 0) has zero derivatives.
        """
        dt = t0 - t  # note: reversed convention compared to Bazin/Sigmoid
        jac = np.zeros((3, len(dt)))

        # Skip the near-zero rise_time regime where `value` is clipped via
        # `protected_rise`; the gradient there is meaningless anyway.
        if abs(rise_time) < 1e-5:
            return jac

        tau = rise_time
        e_const = math.e
        # Active wherever the raw expression is positive (i.e. dt and τ same sign).
        active = (np.sign(dt) == np.sign(tau)) & (dt != 0.0)
        if not np.any(active):
            return jac

        dt_a = dt[active]
        exp_pow = np.exp(-dt_a / tau)
        coeff = amplitude * e_const / tau

        # ∂f/∂t0  = A·e·E/τ · (1 - dt/τ)
        jac[0, active] = coeff * exp_pow * (1.0 - dt_a / tau)
        # ∂f/∂A   = e · dt · E / τ
        jac[1, active] = e_const * dt_a * exp_pow / tau
        # ∂f/∂τ   = A·e·dt·E · (dt - τ) / τ³
        jac[2, active] = coeff * dt_a * exp_pow * (dt_a - tau) / (tau * tau)

        return jac


@dataclass()
class DoublexpBolometricTerm(BaseBolometricTerm):
    """Doublexp function generated using Multi-view Symbolic Regression on ZTF SNIa light curves
    Russeil et al. 2024, https://arxiv.org/abs/2402.04298"""

    @staticmethod
    def parameter_names():
        return ["reference_time", "amplitude", "time1", "time2", "p"]

    @staticmethod
    def parameter_scalings():
        return ["time", "flux", "timescale", "timescale", "None"]

    @staticmethod
    def value(t, t0, amplitude, time1, time2, p):
        dt = t - t0
        result = np.zeros_like(dt)

        # To avoid numerical overflows
        maxp = 20
        A = -(dt / time1) * (p - np.exp(-(dt / time2)))
        A = np.where(A > maxp, maxp, A)

        result = amplitude * np.exp(A)

        return result

    @staticmethod
    def initial_guesses(t, m, sigma, band):
        A = max(np.max(m), np.ptp(m))
        t0, dt = t0_and_weighted_centroid_sigma(t, m, sigma)

        # Empirical conversion of sigma to times
        time1 = 2 * dt
        time2 = 2 * dt

        initial = {}
        initial["reference_time"] = t0
        initial["amplitude"] = A
        initial["time1"] = time1
        initial["time2"] = time2
        initial["p"] = 1

        return initial

    @staticmethod
    def limits(t, m, sigma, band):
        t_amplitude = np.ptp(t)
        m_amplitude = np.ptp(m)
        _, dt = t0_and_weighted_centroid_sigma(t, m, sigma)

        limits = {}
        limits["reference_time"] = (np.min(t) - 10 * t_amplitude, np.max(t) + 10 * t_amplitude)
        limits["amplitude"] = (0.0, 10 * m_amplitude)
        limits["time1"] = (dt / 10, 2 * t_amplitude)
        limits["time2"] = (dt / 10, 2 * t_amplitude)
        limits["p"] = (1e-2, 100)

        return limits

    @staticmethod
    def peak_time(t0, p):
        try:
            from scipy.special import lambertw
        except ImportError:
            raise ImportError("scipy is required for DoublexpBolometricTerm.peak_time, please install it")

        return t0 + np.real(-lambertw(p * np.exp(1)) + 1)

    @staticmethod
    def derivatives(t, t0, amplitude, time1, time2, p):
        """Jacobian of `value` w.r.t. (t0, amplitude, time1, time2, p), shape (5, len(t)).

        Where the inner exponent is clamped (A > maxp=20), the model becomes
        constant in (t0, time1, time2, p), so only the amplitude derivative is
        non-zero there.
        """
        dt = t - t0
        n = len(dt)
        jac = np.zeros((5, n))

        v = np.exp(-dt / time2)
        a_inner = -(dt / time1) * (p - v)
        maxp = 20.0
        clamped = a_inner > maxp
        active = ~clamped

        # B = amplitude · exp(min(a_inner, maxp))
        B = amplitude * np.exp(np.where(clamped, maxp, a_inner))

        # ∂B/∂amplitude = B / amplitude (always true; in clamped region too)
        jac[1] = B / amplitude

        if np.any(active):
            dt_a = dt[active]
            v_a = v[active]
            B_a = B[active]
            # ∂a_inner/∂t0  = (p - v)/τ1 + dt · v / (τ1 · τ2)
            da_dt0 = (p - v_a) / time1 + dt_a * v_a / (time1 * time2)
            # ∂a_inner/∂τ1  = dt · (p - v) / τ1²
            da_dτ1 = dt_a * (p - v_a) / (time1 * time1)
            # ∂a_inner/∂τ2  = dt² · v / (τ1 · τ2²)
            da_dτ2 = (dt_a * dt_a) * v_a / (time1 * time2 * time2)
            # ∂a_inner/∂p   = -dt / τ1
            da_dp = -dt_a / time1

            jac[0, active] = B_a * da_dt0
            jac[2, active] = B_a * da_dτ1
            jac[3, active] = B_a * da_dτ2
            jac[4, active] = B_a * da_dp

        return jac


def median_dt(t, band):
    # Compute the median distance between points in each band
    # Caution when using this method as it might be strongly biaised because of ZTF high cadence a given day.
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


bolometric_terms = {
    "sigmoid": SigmoidBolometricTerm,
    "bazin": BazinBolometricTerm,
    "linexp": LinexpBolometricTerm,
    "doublexp": DoublexpBolometricTerm,
}
