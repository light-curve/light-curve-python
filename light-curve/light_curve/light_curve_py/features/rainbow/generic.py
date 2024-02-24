from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np

from light_curve.light_curve_py.dataclass_field import dataclass_field
from light_curve.light_curve_py.features.rainbow._base import BaseRainbowFit
from light_curve.light_curve_py.features.rainbow._scaler import MultiBandScaler, Scaler

from .bolometric import BaseBolometricTerm, bolometric_terms
from .temperature import BaseTemperatureTerm, temperature_terms

__all__ = ["RainbowFit"]

# CODATA 2018, grab from astropy
planck_constant = 6.62607004e-27  # erg s
speed_of_light = 2.99792458e10  # cm/s
boltzman_constant = 1.380649e-16  # erg/K
sigma_sb = 5.6703744191844314e-05  # erg/(cm^2 s K^4)


@dataclass()
class RainbowFit(BaseRainbowFit):
    """Multiband blackbody fit to the light curve using functions to be chosen by the user
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
    peak_time(*params)
        Return bolometric peak time for given set of parameters
    """

    bolometric: Union[str, BaseBolometricTerm] = dataclass_field(default="bazin", kw_only=True)
    """Which parametric bolometric term to use"""
    temperature: Union[str, BaseTemperatureTerm] = dataclass_field(default="sigmoid", kw_only=True)
    """Which parametric temperature term to use"""

    def __post_init__(self):
        if not isinstance(self.bolometric, BaseBolometricTerm):
            self.bolometric = bolometric_terms[self.bolometric]

        if not isinstance(self.temperature, BaseTemperatureTerm):
            self.temperature = temperature_terms[self.temperature]

        super().__post_init__()

    def _common_parameter_names(self) -> List[str]:
        bolometric_parameters = self.bolometric.parameter_names()
        temperature_parameters = self.temperature.parameter_names()
        return [j for j in bolometric_parameters if j in temperature_parameters]

    def _bolometric_parameter_names(self) -> List[str]:
        bolometric_parameters = self.bolometric.parameter_names()
        return [i for i in bolometric_parameters if i not in self._common_parameter_names()]

    def _temperature_parameter_names(self) -> List[str]:
        temperature_parameters = self.temperature.parameter_names()
        return [i for i in temperature_parameters if i not in self._common_parameter_names()]

    def bol_func(self, t, params):
        return self.bolometric.value(t, *params[self.p.all_bol_idx])

    def temp_func(self, t, params):
        return self.temperature.value(t, *params[self.p.all_temp_idx])

    def _unscale_parameters(self, params, t_scaler: Scaler, m_scaler: MultiBandScaler, scale_errors=False) -> None:
        already_unscaled = {}
        for term in [self.bolometric, self.temperature]:
            for name,scaling in zip(term.parameter_names(), term.parameter_scalings()):
                if name in already_unscaled:
                    # Avoid un-scaling common parametres twice
                    continue

                if scaling == "time":
                    if scale_errors:
                        params[self.p[name]] = t_scaler.undo_scale(params[self.p[name]])
                    else:
                        params[self.p[name]] = t_scaler.undo_shift_scale(params[self.p[name]])

                elif scaling == "timescale":
                    params[self.p[name]] = t_scaler.undo_scale(params[self.p[name]])

                elif scaling == "flux":
                    params[self.p[name]] = m_scaler.undo_scale(params[self.p[name]])

                already_unscaled[name] = True

    def _unscale_errors(self, errors, t_scaler: Scaler, m_scaler: MultiBandScaler) -> None:
        self._unscale_parameters(errors, t_scaler, m_scaler, scale_errors=True)

    def _initial_guesses(self, t, m, band) -> Dict[str, float]:
        initial_bolometric = self.bolometric.initial_guesses(t, m, band)
        initial_temp = self.temperature.initial_guesses(t, m, band)

        return initial_bolometric | initial_temp

    def _limits(self, t, m, band) -> Dict[str, Tuple[float, float]]:
        limits_bolometric = self.bolometric.limits(t, m, band)
        limits_temp = self.temperature.limits(t, m, band)

        return limits_bolometric | limits_temp

    def peak_time(self, params) -> float:
        """Returns true bolometric peak position for given parameters"""
        return self.bolometric.peak_time(*params[self.p.all_bol_idx])
