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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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
