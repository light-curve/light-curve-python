from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

from light_curve.light_curve_py.dataclass_field import dataclass_field
from light_curve.light_curve_py.features.rainbow._base import BaseRainbowFit

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
    bolometric : str or BaseBolometricTerm subclass, optional
        The shape of bolometric term. Default is 'bazin'.
        Other options are: 'sigmoid'
    temperature : str or BaseTemperatureTerm subclass, optional
        The shape of temperature term. Default is 'sigmoid'.
        Other options are: 'constant', 'delayed_sigmoid'

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
    fit_and_get_errors(t, m, sigma, band, sorted=False, check=True, print_level=None, get_initial=False)
        The same as `__call__` but also returns the parameter errors. Optionally sets the `print_level`
        (verbosity) for Minuit fitter. If `get_initial` is True, returns the initial parameters instead
        of fitted ones (useful for debugging)
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

    def _parameter_scalings(self) -> Dict[str, str]:
        rules = super()._parameter_scalings()

        for term in [self.bolometric, self.temperature]:
            for name, scaling in zip(term.parameter_names(), term.parameter_scalings()):
                rules[name] = scaling

        return rules

    def _initial_guesses(self, t, m, sigma, band) -> Dict[str, float]:
        initial = self.bolometric.initial_guesses(t, m, sigma, band)
        initial.update(self.temperature.initial_guesses(t, m, sigma, band))

        return initial

    def _limits(self, t, m, sigma, band) -> Dict[str, Tuple[float, float]]:
        limits = self.bolometric.limits(t, m, sigma, band)
        limits.update(self.temperature.limits(t, m, sigma, band))

        return limits

    def peak_time(self, params) -> float:
        """Returns true bolometric peak position for given parameters"""
        return self.bolometric.peak_time(*params[self.p.all_bol_idx])
