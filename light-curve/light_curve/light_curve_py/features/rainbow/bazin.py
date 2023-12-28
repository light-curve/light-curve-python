from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from light_curve.light_curve_py.features.rainbow._base import BaseRainbowFit
from light_curve.light_curve_py.features.rainbow._scaler import MultiBandScaler, Scaler

__all__ = ["RainbowFit"]


@dataclass()
class RainbowFit(BaseRainbowFit):
    """Multiband blackbody fit to the light curve.

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
    """

    @staticmethod
    def _common_parameter_names() -> List[str]:
        return ["reference_time"]

    @staticmethod
    def _bolometric_parameter_names() -> List[str]:
        return ["amplitude", "rise_time", "fall_time"]

    @staticmethod
    def _temperature_parameter_names() -> List[str]:
        return ["Tmin", "delta_T", "k_sig"]

    def bol_func(self, t, params):
        t0, amplitude, rise_time, fall_time = params[self.p.all_bol_idx]
        dt = t - t0
        return amplitude * np.exp(-dt / fall_time) / (1.0 + np.exp(-dt / rise_time))

    def temp_func(self, t, params):
        t0, T_min, delta_T, k_sig = params[self.p.all_temp_idx]
        dt = t - t0
        return T_min + delta_T / (1.0 + np.exp(dt / k_sig))

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

        amplitude, rise_time, fall_time = params[self.p.bol_idx]
        amplitude = m_scaler.undo_scale(amplitude)
        rise_time = t_scaler.undo_scale(rise_time)
        fall_time = t_scaler.undo_scale(fall_time)
        params[self.p.bol_idx] = amplitude, rise_time, fall_time

        T_min, delta_T, k_sig = params[self.p.temp_idx]
        k_sig = t_scaler.undo_scale(k_sig)
        params[self.p.temp_idx] = T_min, delta_T, k_sig

    def _initial_guesses(self, t, m, band) -> Dict[str, float]:
        del t, m, band
        return {
            "reference_time": 0.0,
            "amplitude": 1.0,
            "rise_time": 1.0,
            "fall_time": 1.0,
            "Tmin": 4000.0,
            "delta_T": 7000.0,
            "k_sig": 1.0,
        }

    def _limits(self, t, m, band) -> Dict[str, Tuple[float, float]]:
        del band

        t_amplitude = t[-1] - t[0]
        m_amplitude = np.ptp(m)
        return {
            "reference_time": (t[0] - 10 * t_amplitude, t[-1] + 10 * t_amplitude),
            "amplitude": (0.0, 10 * m_amplitude),
            "rise_time": (0.0, 10 * t_amplitude),
            "fall_time": (0.0, 10 * t_amplitude),
            "Tmin": (1e2, 1e6),  # K
            "delta_T": (0.0, 1e6),  # K
            "k_sig": (0.0, 10 * t_amplitude),
        }
