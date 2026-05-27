from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np

__all__ = [
    "spectral_terms",
    "BaseSpectralTerm",
    "PlanckSpectralTerm",
    "BlanketedPlanckConstTemp",
    "BlanketedPlanckSigmoidTemp",
]

# CODATA 2018
planck_constant = 6.62607004e-27  # erg s
speed_of_light = 2.99792458e10  # cm/s
boltzman_constant = 1.380649e-16  # erg/K
b_wien = 28977720  # Angstrom*K


@dataclass()
class BaseSpectralTerm:
    """Spectral term for Rainbow"""

    @staticmethod
    @abstractmethod
    def parameter_names() -> List[str]:
        return NotImplementedError

    @staticmethod
    @abstractmethod
    def parameter_scalings() -> List[Union[str, None]]:
        return NotImplementedError

    @staticmethod
    @abstractmethod
    def value(wave_cm, T, *params):
        return NotImplementedError

    @staticmethod
    @abstractmethod
    def initial_guesses(t, m, sigma, band) -> Dict[str, float]:
        return NotImplementedError

    @staticmethod
    @abstractmethod
    def limits(t, m, sigma, band) -> Dict[str, float]:
        return NotImplementedError


@dataclass()
class PlanckSpectralTerm(BaseSpectralTerm):
    """Standard blackbody spectrum"""

    @staticmethod
    def parameter_names():
        return []

    @staticmethod
    def parameter_scalings():
        return []

    @staticmethod
    def value(wave_cm, T, *params):
        nu = speed_of_light / wave_cm

        return (
            (2 * planck_constant / speed_of_light**2) * nu**3 / np.expm1(planck_constant * nu / (boltzman_constant * T))
        )

    @staticmethod
    def initial_guesses(t, m, sigma, band):
        return {}

    @staticmethod
    def limits(t, m, sigma, band):
        return {}

    @staticmethod
    def dvalue_dT(wave_cm, T, *params):
        """∂(planck)/∂T. `*params` is empty for plain Planck (kept for API parity)."""
        nu = speed_of_light / wave_cm
        x = planck_constant * nu / (boltzman_constant * T)
        em1 = np.expm1(x)
        # planck = (2h/c²) ν³ / expm1(x);   x = hν/(kT);   dx/dT = -x/T
        # ∂planck/∂T = planck · x·exp(x) / (T · expm1(x))
        planck = (2 * planck_constant / speed_of_light**2) * nu**3 / em1
        return planck * x * np.exp(x) / (T * em1)

    @staticmethod
    def derivatives(wave_cm, T, *params):
        """No spectral parameters; returns shape (0, len(wave_cm))."""
        return np.zeros((0, len(wave_cm)))


@dataclass()
class BlanketedPlanckConstTemp(BaseSpectralTerm):
    """Blackbody spectrum with exponential blanketing, anchored to a constant temperature.

    Pairs with ``temperature='constant'``: the extinction reach scales as ``1/T`` with the
    (constant-in-time) temperature, so the blanketing depth does not vary over the light
    curve. See :class:`BlanketedPlanckSigmoidTemp` for the cooling-source variant.
    """

    # Fixed blanketing intensity and extinction reach; referenced by `value` and the gradients.
    _intensity = 100
    _max_extinction = 2 * b_wien

    @staticmethod
    def parameter_names():
        return ["lambda_scale"]

    @staticmethod
    def parameter_scalings():
        return [None]

    @staticmethod
    def value(wave_cm, T, lambda_scale):
        base = PlanckSpectralTerm.value(wave_cm, T)

        # Phenomenological value for the slope of the extinction
        # Fitting instead of fixing is likely overkill for broad band photometry
        intensity = BlanketedPlanckConstTemp._intensity

        # Encodes how far (in wavelength) the maximum UV extinction extends (at lambda_scale=1).
        # Allows the extinction to affect the BB wavelength past the peak (in the formula below).
        max_extinction = BlanketedPlanckConstTemp._max_extinction

        # Lambda_angstrom represents how far (in absolute wavelength) the extinction affects the BB.
        # Lambda_scale quantifies this between 0 (no UV ext) and 1 max suppression (encoded by max_extinction above)
        # We want this maximum extinction to scale with the size of the BB.
        # We use T to give the correct order of magntiude of the scaling
        lambda_angstrom = max_extinction * lambda_scale / T

        # Convert to cm to match wave_cm
        lambda_cm = lambda_angstrom * 1e-8

        tau = intensity * np.exp(-wave_cm / lambda_cm)

        return base * np.exp(-tau)

    @staticmethod
    def initial_guesses(t, m, sigma, band):
        return {
            "lambda_scale": 0.001,
        }

    @staticmethod
    def limits(t, m, sigma, band):
        return {
            "lambda_scale": (0.001, 1.0),
        }

    @staticmethod
    def _planck_and_tau(wave_cm, T, lambda_scale):
        nu = speed_of_light / wave_cm
        x = planck_constant * nu / (boltzman_constant * T)
        em1 = np.expm1(x)
        planck = (2 * planck_constant / speed_of_light**2) * nu**3 / em1
        # Extinction reach scales inversely with temperature (see `value`).
        lambda_cm = BlanketedPlanckConstTemp._max_extinction * lambda_scale / T * 1e-8
        u = wave_cm / lambda_cm
        tau = BlanketedPlanckConstTemp._intensity * np.exp(-u)
        return planck, em1, x, tau, u

    @staticmethod
    def dvalue_dT(wave_cm, T, lambda_scale):
        """∂(spec)/∂T, including the temperature dependence of the extinction τ(T)."""
        planck, em1, x, tau, u = BlanketedPlanckConstTemp._planck_and_tau(wave_cm, T, lambda_scale)
        exp_mtau = np.exp(-tau)
        dplanck_dT = planck * x * np.exp(x) / (T * em1)
        spec = planck * exp_mtau
        # τ = I·e^{-u},  u = wave_cm/λ_cm,  λ_cm ∝ λ_scale/T  ⇒  ∂u/∂T = u/T,  ∂τ/∂T = -τ·u/T
        # ∂spec/∂T = e^{-τ}·∂planck/∂T - spec·∂τ/∂T = e^{-τ}·∂planck/∂T + spec·τ·u/T
        return dplanck_dT * exp_mtau + spec * tau * u / T

    @staticmethod
    def derivatives(wave_cm, T, lambda_scale):
        """∂(spec)/∂lambda_scale; shape (1, len(wave_cm)). Intensity is fixed, not fitted."""
        planck, _em1, _x, tau, u = BlanketedPlanckConstTemp._planck_and_tau(wave_cm, T, lambda_scale)
        spec = planck * np.exp(-tau)
        # u = wave_cm/λ_cm ∝ 1/λ_scale  ⇒  ∂u/∂λ_scale = -u/λ_scale,  ∂τ/∂λ_scale = τ·u/λ_scale
        # ∂spec/∂λ_scale = -spec · ∂τ/∂λ_scale
        jac = np.zeros((1, len(wave_cm)))
        jac[0] = -spec * tau * u / lambda_scale
        return jac


@dataclass()
class BlanketedPlanckSigmoidTemp(BaseSpectralTerm):
    """Blackbody spectrum with exponential blanketing."""

    @staticmethod
    def parameter_names():
        return ["Tmin", "Tmax", "lambda_scale"]

    @staticmethod
    def parameter_scalings():
        return [None, None, None]

    @staticmethod
    def value(wave_cm, T, Tmin, Tmax, lambda_scale):
        base = PlanckSpectralTerm.value(wave_cm, T)

        # Phenomenological value for the slope of the extinction
        # Fitting instead of fixing is likely overkill for broad band photometry
        intensity = 100

        # Encodes how far (in wavelength) the maximum UV extinction extends (at lambda_scale=1).
        # Allows the extinction to affect the BB wavelength past the peak (in the formula below).
        max_extinction = 2 * b_wien

        # The depth of exinction depends on the temperature of source.
        # But it should not vary as the object cools down. We use T(t=~peak)
        T_scale = (Tmin + Tmax) / 2

        # Lambda_angstrom represents how far (in absolute wavelength) the extinction affects the BB.
        # Lambda_scale quantifies this between 0 (no UV ext) and 1 max suppression (encoded by max_extinction above)
        lambda_angstrom = max_extinction * lambda_scale / T_scale

        # Convert to cm to match wave_cm
        lambda_cm = lambda_angstrom * 1e-8

        tau = intensity * np.exp(-wave_cm / lambda_cm)

        return base * np.exp(-tau)

    @staticmethod
    def initial_guesses(t, m, sigma, band):
        return {
            "lambda_scale": 0.001,
        }

    @staticmethod
    def limits(t, m, sigma, band):
        return {
            "lambda_scale": (0.001, 1.0),
        }


spectral_terms = {
    "planck": PlanckSpectralTerm,
    "blanketed_constant_temperature": BlanketedPlanckConstTemp,
    "blanketed_sigmoid_temperature": BlanketedPlanckSigmoidTemp,
}
