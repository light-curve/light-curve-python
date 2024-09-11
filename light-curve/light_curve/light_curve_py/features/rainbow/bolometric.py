import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
from scipy.special import lambertw

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
        return ["reference_time", "rise_time", "amplitude"]

    @staticmethod
    def parameter_scalings():
        return ["time", "timescale", "flux"]

    @staticmethod
    def value(t, t0, rise_time, amplitude):
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
        initial["rise_time"] = 1.0 if m[0] < m[-1] else -1

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
    def peak_time(t0, rise_time, amplitude):
        """Peak time is not defined for the sigmoid, so it returns mid-time of the rise instead"""
        return t0


@dataclass()
class BazinBolometricTerm(BaseBolometricTerm):
    """Bazin function, symmetric form"""

    @staticmethod
    def parameter_names():
        return ["reference_time", "rise_time", "amplitude", "fall_time"]

    @staticmethod
    def parameter_scalings():
        return ["time", "timescale", "flux", "timescale"]

    @staticmethod
    def value(t, t0, rise_time, amplitude, fall_time):
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

        print(f"Guess: [{rise_time}]")

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

        print(f"Limits: [{1e-4}, {10 * t_amplitude}]")

        return limits

    @staticmethod
    def peak_time(t0, rise_time, amplitude, fall_time):
        return t0 + np.log(fall_time / rise_time) * rise_time * fall_time / (rise_time + fall_time)


@dataclass()
class ExpBolometricTerm(BaseBolometricTerm):
    """Exp function"""

    @staticmethod
    def parameter_names():
        return ["pseudo_amplitude", "rise_time"]

    @staticmethod
    def parameter_scalings():
        # In reality the fake_pseudo_amplitude parameter here is not in flux unit.
        # Because without t0 the exponential is not dimentionless
        # However in the special case were mean(t) = 0, then everything works.
        # We use this trick for this specific function
        return ["flux", "timescale"]

    @staticmethod
    def value(t, rise_time, pseudo_amplitude):
        protected = np.where(t / rise_time > 10, 10, t / rise_time)
        return pseudo_amplitude * np.exp(protected)

    @staticmethod
    def initial_guesses(t, m, sigma, band):
        initial = {}
        initial["rise_time"] = 10 if m[0] < m[-1] else -10
        initial["pseudo_amplitude"] = 1

        return initial

    @staticmethod
    def limits(t, m, sigma, band):
        t_amplitude = np.ptp(t)

        limits = {}
        limits["rise_time"] = (-10 * t_amplitude, 10 * t_amplitude)
        limits["pseudo_amplitude"] = (0.0, 10 * t_amplitude)  #

        return limits

    @staticmethod
    def peak_time(A, rise_time):
        return rise_time * (10 - np.log(A))


@dataclass()
class LinexpBolometricTerm(BaseBolometricTerm):
    """Linexp function, symmetric form. Generated using a prototype version of Multi-view
    Symbolic Regression (Russeil et al. 2024, https://arxiv.org/abs/2402.04298) on
    a SLSN ZTF light curve (https://ztf.snad.space/dr17/view/821207100004043)"""

    @staticmethod
    def parameter_names():
        return ["reference_time", "rise_time", "amplitude"]

    @staticmethod
    def parameter_scalings():
        return ["time", "timescale", "flux"]

    @staticmethod
    def value(t, t0, rise_time, amplitude):
        dt = t0 - t
        protected_rise = math.copysign(max(1e-5, abs(rise_time)), rise_time)

        # Coefficient to make peak amplitude equal to unity
        scale = 1 / (protected_rise * np.exp(-1))

        power = -(math.copysign(1, amplitude) * dt) / protected_rise
        power = np.where(power > 100, 100, power)
        result = amplitude * scale * dt * np.exp(power)

        return result

    @staticmethod
    def initial_guesses(t, m, sigma, band):

        A = np.max(m)

        # Compute points after or before maximum
        peak_time = t[np.argmax(m)]
        after = t[-1] - peak_time
        before = peak_time - t[0]

        t0 = t[-1] if before >= after else t[0]

        # Peak position as weighted centroid of everything above zero
        idx = m > 0
        # Weighted centroid sigma
        dt = np.sqrt(np.sum((t[idx] - t0) ** 2 * m[idx] / sigma[idx]) / np.sum(m[idx] / sigma[idx]))
        # Empirical conversion of sigma to rise/rise times
        rise_time = dt / 2

        initial = {}
        initial["reference_time"] = t0
        initial["rise_time"] = rise_time if before >= after else -rise_time
        initial["amplitude"] = A

        return initial

    @staticmethod
    def limits(t, m, sigma, band):
        t_amplitude = np.ptp(t)
        m_amplitude = np.max(m)

        limits = {}
        limits["reference_time"] = (np.min(t) - 10 * t_amplitude, np.max(t) + 10 * t_amplitude)
        limits["rise_time"] = (-10 * t_amplitude, 10 * t_amplitude)
        limits["amplitude"] = (0, 10 * m_amplitude)

        return limits

    @staticmethod
    def peak_time(t0, rise_time, amplitude):
        return t0 - rise_time


@dataclass()
class DoublexpBolometricTerm(BaseBolometricTerm):
    """Doublexp function generated using Multi-view Symbolic Regression on ZTF SNIa light curves
    Russeil et al. 2024, https://arxiv.org/abs/2402.04298"""

    @staticmethod
    def parameter_names():
        return ["reference_time", "time1", "amplitude", "time2", "p"]

    @staticmethod
    def parameter_scalings():
        return ["time", "timescale", "flux", "timescale", "None"]

    @staticmethod
    def value(t, t0, time1, amplitude, time2, p):
        dt = t - t0

        result = np.zeros_like(dt)

        ## To avoid numerical overflows
        maxp = 20
        A = -(dt / time1) * (p - np.exp(-(dt / time2)))
        A = np.where(A > maxp, maxp, A)

        result = amplitude * np.exp(A)

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
        time1 = 10 * dt
        time2 = 10 * dt

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
        m_amplitude = np.max(m)

        limits = {}
        limits["reference_time"] = (np.min(t) - 10 * t_amplitude, np.max(t) + 10 * t_amplitude)
        limits["amplitude"] = (0.0, 10 * m_amplitude)
        limits["time1"] = (1e-1, 2 * t_amplitude)
        limits["time2"] = (1e-1, 2 * t_amplitude)
        limits["p"] = (1e-2, 100)

        return limits

    @staticmethod
    def peak_time(t0, p):
        return t0 + np.real(-lambertw(p * np.exp(1)) + 1)


bolometric_terms = {
    "sigmoid": SigmoidBolometricTerm,
    "bazin": BazinBolometricTerm,
    "exp": ExpBolometricTerm,
    "linexp": LinexpBolometricTerm,
    "doublexp": DoublexpBolometricTerm,
}
