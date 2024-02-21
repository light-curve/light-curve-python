from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from light_curve.light_curve_py.dataclass_field import dataclass_field
from light_curve.light_curve_py.features.rainbow._base import BaseRainbowFit
from light_curve.light_curve_py.features.rainbow._scaler import MultiBandScaler, Scaler

__all__ = ["RainbowSymmetricFit"]


@dataclass()
class RainbowSymmetricFit(BaseRainbowFit):
    """Multiband blackbody fit to the light curve using symmetric form of Bazin function

    Note, that `m` and corresponded `sigma` are assumed to be flux densities.

    Based on Russeil et al. 2023, arXiv:2310.02916.

    Parameters
    ----------
    band_wave_cm : dict
        Dictionary of band names and their effective wavelengths in cm.

    with_baseline : bool, optional
        Whether to include an offset in the fit, individual for each band.
        If it is true, one more fit paramter per passband is added -
        the additive constant with the same units as input flux.

    with_temperature_evolution : bool, optional
        Whether to include temperature evolution in the fit

    with_rising_only : bool, optional
        If false, only rising part of the Bazin function is used, making
        it effectively a sigmoid function

    Methods
    -------
    __call__(t, m, sigma, band, *, sorted=False, check=True, fill_value=None)
        Evaluate the feature. Positional arguments are numpy arrays of the same length,
        `band` must consist of the same strings as keys in `band_wave_cm`. If `sorted` is True,
        `t` must be sorted in ascending order. If `check` is True, the input is checked for
        NaNs and Infs. If `fill_value` is not None, it is used to fill the output array if
        the feature cannot be evaluated.

    model(t, band, *params)
        Evaluate Rainbow model on the given arrays of times and bands. `*params` are
        fit parameters, basically the output of `__call__` method but without the last
        parameter (reduced Chi^2 of the fit). See parameter names in the `.name` attribute.

    peak_time(*params)
        Return bolometric peak time for given set of parameters
    """

    with_temperature_evolution: bool = dataclass_field(default=True, kw_only=True)
    """Whether to include a temperature evolution in the fit."""
    with_rise_only: bool = dataclass_field(default=False, kw_only=True)
    """Whether to use sigmoid (rising part only) bolometric function in the fit."""

    @staticmethod
    def _common_parameter_names() -> List[str]:
        return ["reference_time"]

    def _bolometric_parameter_names(self) -> List[str]:
        if self.with_rise_only:
            return ["amplitude", "rise_time"]
        else:
            return ["amplitude", "rise_time", "fall_time"]

    def _temperature_parameter_names(self) -> List[str]:
        if self.with_temperature_evolution:
            return ["Tmin", "Tmax", "k_sig"]
        else:
            return ["Tmin"]

    def bol_func(self, t, params):
        if self.with_rise_only:
            t0, amplitude, rise_time = params[self.p.all_bol_idx]
        else:
            t0, amplitude, rise_time, fall_time = params[self.p.all_bol_idx]
        dt = t - t0
        result = np.zeros_like(dt)

        # To avoid numerical overflows, let's only compute the exponents not too far from t0
        if self.with_rise_only:
            idx = dt > -100 * rise_time
            result[idx] = amplitude / (np.exp(-dt[idx] / rise_time) + 1)
        else:
            idx = (dt > -100 * rise_time) & (dt < 100 * fall_time)
            result[idx] = amplitude / (np.exp(-dt[idx] / rise_time) + np.exp(dt[idx] / fall_time))

        return result

    def temp_func(self, t, params):
        if self.with_temperature_evolution:
            t0, T_min, T_max, k_sig = params[self.p.all_temp_idx]
            dt = t - t0

            result = np.zeros_like(dt)

            # To avoid numerical overflows, let's only compute the exponent not too far from t0
            idx1 = dt <= -100 * k_sig
            idx2 = (dt > -100 * k_sig) & (dt < 100 * k_sig)
            idx3 = dt >= 100 * k_sig

            result[idx1] = T_min
            result[idx2] = T_min + (T_max - T_min) / (1.0 + np.exp(dt[idx2] / k_sig))
            result[idx3] = T_max

            return result

        t0, T_min = params[self.p.all_temp_idx]

        return T_min

    def _normalize_bolometric_flux(self, params) -> None:
        # Internally we use amplitude of F_bol / <nu> instead of F_bol.
        # It makes amplitude to be in the same units and the same order as
        # the baselines and input fluxes.
        params[self.p.amplitude] /= self.average_nu

    def _denormalize_bolometric_flux(self, params) -> None:
        params[self.p.amplitude] *= self.average_nu

    def _unscale_parameters(self, params, t_scaler: Scaler, m_scaler: MultiBandScaler) -> None:
        self._denormalize_bolometric_flux(params)

        t0 = params[self.p.reference_time]
        t0 = t_scaler.undo_shift_scale(t0)
        params[self.p.reference_time] = t0

        if self.with_rise_only:
            amplitude, rise_time = params[self.p.bol_idx]
            amplitude = m_scaler.undo_scale(amplitude)
            rise_time = t_scaler.undo_scale(rise_time)
            params[self.p.bol_idx] = amplitude, rise_time
        else:
            amplitude, rise_time, fall_time = params[self.p.bol_idx]
            amplitude = m_scaler.undo_scale(amplitude)
            rise_time = t_scaler.undo_scale(rise_time)
            fall_time = t_scaler.undo_scale(fall_time)
            params[self.p.bol_idx] = amplitude, rise_time, fall_time

        if self.with_temperature_evolution:
            T_min, T_max, k_sig = params[self.p.temp_idx]
            k_sig = t_scaler.undo_scale(k_sig)
            params[self.p.temp_idx] = T_min, T_max, k_sig

    def _initial_guesses(self, t, m, band) -> Dict[str, float]:
        if self.with_rise_only:
            t_rise = 1.0
        else:
            t_rise = 0.1
        t_fall = 0.1

        # The amplitude here does not actually correspond to the amplitude of m values!
        if self.with_baseline:
            A = np.ptp(m)
        else:
            A = np.max(m)

        params = {}

        # Why do we have to strictly follow the order of parameters here?..
        params["reference_time"] = t[np.argmax(m)]
        params["amplitude"] = A
        params["rise_time"] = t_rise

        if not self.with_rise_only:
            params["fall_time"] = t_fall

        params["Tmin"] = 7000.0

        if self.with_temperature_evolution:
            params["Tmax"] = 10000.0
            params["k_sig"] = 1.0

        return params

    def _limits(self, t, m, band) -> Dict[str, Tuple[float, float]]:
        t_amplitude = np.ptp(t)
        if self.with_baseline:
            m_amplitude = np.ptp(m)
        else:
            m_amplitude = np.max(m)

        limits = {}

        # Why do we have to strictly follow the order of parameters here?..
        limits["reference_time"] = (np.min(t) - 10 * t_amplitude, np.max(t) + 10 * t_amplitude)
        limits["amplitude"] = (0.0, 10 * m_amplitude)
        limits["rise_time"] = (1e-4, 10 * t_amplitude)

        if not self.with_rise_only:
            limits["fall_time"] = (1e-4, 10 * t_amplitude)

        limits["Tmin"] = (1e2, 1e6)  # K

        if self.with_temperature_evolution:
            limits["Tmax"] = (1e2, 1e6)  # K
            limits["k_sig"] = (1e-4, 10.0 * t_amplitude)

        return limits

    def _baseline_initial_guesses(self, t, m, band) -> Dict[str, float]:
        """Initial guesses for the baseline parameters."""
        return {self.p.baseline_parameter_name(b): np.median(m[band == b]) for b in self.bands.names}

    def peak_time(self, params) -> float:
        """Returns true bolometric peak position for given parameters"""
        if self.with_rise_only:
            t0, amplitude, rise_time = params[self.p.all_bol_idx]

            # It is not, strictly speaking, defined for rising only
            return t0

        t0, amplitude, rise_time, fall_time = params[self.p.all_bol_idx]

        return t0 + np.log(fall_time / rise_time) * rise_time * fall_time / (rise_time + fall_time)
