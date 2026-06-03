from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np

__all__ = [
    "spectral_terms",
    "BaseSpectralTerm",
    "PlanckSpectralTerm",
    "BlanketedPlanckConstTemp",
    "BlanketedPlanckSigmoidTemp",
    "GenWienSpectralTerm",
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

    @staticmethod
    def parameter_priors() -> Dict[str, Tuple[float, float]]:
        """Optional Gaussian priors ``{name: (mean, sigma)}`` on parameters.

        Only parameters with ``None`` scaling are supported (the prior is applied
        directly in the fit's scaled space, which for unscaled parameters equals the
        physical value). Default: no priors.
        """
        return {}


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


@dataclass()
class GenWienSpectralTerm(BaseSpectralTerm):
    r"""Generalized-Wien optical SED.

    .. math::
        B(\nu) \propto \nu^3 \, e^{-x^{\texttt{spec\_k}}},\quad x = \frac{h\nu}{k_B T}

    A single parameter ``spec_k`` (the blue-falloff exponent) unifies blackbody-like
    and UV-blanketed optical SEDs in one smooth form, replacing the separate
    Planck × extinction product:

    - ``spec_k ≈ 1`` is the Wien tail, which equals Planck for cool sources (large ``x``);
    - ``spec_k > 1`` sharpens the blue cutoff, reproducing the UV deficit of hot sources.

    Unlike the blanketed model, the time-variation of the blue suppression is *inherent*
    in the thermal form (it follows ``T(t)`` through ``x``), so ``spec_k`` is a fixed
    per-object shape index rather than a quantity coupled to the instantaneous temperature.

    Cool sources do not constrain ``spec_k`` (it becomes degenerate with ``T``), so a
    Gaussian prior anchors it to the Wien/Planck-like value 1; hot sources, where the data
    do constrain it, override the prior.

    .. warning::
        The fitted ``T`` is **not** a physical (thermodynamic) temperature. Because
        ``spec_k`` and ``T`` trade off, a pure blackbody is recovered at a strongly biased,
        non-monotonic ``T`` (e.g. a 20 kK blackbody fits at ``T ≈ 3 kK``). The bolometric
        normalization (``∝ T⁴``) is correspondingly meaningless, so the recovered amplitude
        is not a physical luminosity. Treat ``(T, spec_k)`` jointly as SED-shape features.

        An approximate blackbody temperature can be recovered from the fitted pair by
        matching the GenWien spectral peak ``ν_peak = (k_B T / h)·(3/spec_k)^(1/spec_k)``
        to the Wien peak of a blackbody:

        .. math::
            T_\mathrm{BB} \approx 0.29 \; T \; (3/\texttt{spec\_k})^{1/\texttt{spec\_k}}

        This is good to ~10-15% only in the warm regime (~5-22 kK). Above ~25 kK both
        ``T`` and ``spec_k`` saturate, so the blackbody temperature is unrecoverable from
        a GenWien fit — use the Planck spectral term if physical temperatures are needed.
    """

    _prior_mean = 1.0
    _prior_sigma = 0.5

    @staticmethod
    def parameter_names():
        return ["spec_k"]

    @staticmethod
    def parameter_scalings():
        return [None]

    @staticmethod
    def _x_value(wave_cm, T, spec_k):
        nu = speed_of_light / wave_cm
        x = planck_constant * nu / (boltzman_constant * T)
        value = (2 * planck_constant / speed_of_light**2) * nu**3 * np.exp(-np.power(x, spec_k))
        return x, value

    @staticmethod
    def value(wave_cm, T, spec_k):
        return GenWienSpectralTerm._x_value(wave_cm, T, spec_k)[1]

    @staticmethod
    def dvalue_dT(wave_cm, T, spec_k):
        """∂(value)/∂T. With ``x = a/T``: ∂x/∂T = -x/T, so ∂value/∂T = value·k·x^k / T."""
        x, value = GenWienSpectralTerm._x_value(wave_cm, T, spec_k)
        return value * spec_k * np.power(x, spec_k) / T

    @staticmethod
    def derivatives(wave_cm, T, spec_k):
        """∂(value)/∂spec_k; shape (1, len(wave_cm)).  ∂value/∂k = -value·x^k·ln(x)."""
        x, value = GenWienSpectralTerm._x_value(wave_cm, T, spec_k)
        jac = np.zeros((1, len(wave_cm)))
        jac[0] = -value * np.power(x, spec_k) * np.log(x)
        return jac

    @staticmethod
    def initial_guesses(t, m, sigma, band):
        return {"spec_k": GenWienSpectralTerm._prior_mean}

    @staticmethod
    def limits(t, m, sigma, band):
        return {"spec_k": (0.3, 3.0)}

    @staticmethod
    def parameter_priors():
        return {"spec_k": (GenWienSpectralTerm._prior_mean, GenWienSpectralTerm._prior_sigma)}


spectral_terms = {
    "planck": PlanckSpectralTerm,
    "blanketed_constant_temperature": BlanketedPlanckConstTemp,
    "blanketed_sigmoid_temperature": BlanketedPlanckSigmoidTemp,
    "genwien": GenWienSpectralTerm,
}
