from dataclasses import dataclass
from enum import IntEnum
from typing import Dict

import numpy as np

from ..dataclass_field import dataclass_field
from ..minuit_lsq import LeastSquares
from ._base import BaseMultiBandFeature

__all__ = ["RainbowFit"]


class P(IntEnum):
    reference_time = 0
    amplitude = 1
    rise_time = 2
    fall_time = 3
    Tmin = 4
    delta_T = 5
    k_sig = 6


# CODATA 2018, grab from astropy
planck_constant = 6.62607004e-27  # erg s
speed_of_light = 2.99792458e10  # cm/s
boltzman_constant = 1.380649e-16  # erg/K
sigma_sb = 5.6703744191844314e-05  # erg/(cm^2 s K^4)


IMINUIT_IMPORT_ERROR = (
    "The `iminuit` package v2.21.0 or larger is required for RainbowFit, "
    "please install it manually or reinstall light-curve package with [full] extra"
)


@dataclass()
class RainbowFit(BaseMultiBandFeature):
    """Multiband blackbody fit to the light curve.

    Note, that `m` and corresponded `sigma` are assumed to be flux densities.

    Based on Russeil et al. 2023, in prep.

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

    band_wave_cm: Dict[str, float]
    """Dictionary of band names and their effective wavelengths in cm."""

    with_baseline: bool = dataclass_field(default=True, kw_only=True)
    """Whether to include a baseline in the fit, one per band."""

    @property
    def is_band_required(self) -> bool:
        return True

    @property
    def is_multiband_supported(self) -> bool:
        return True

    @property
    def size(self) -> int:
        return len(P) + self.with_baseline * len(self.band_wave_cm)

    def __post_init__(self) -> None:
        super().__post_init__()

        self._check_iminuit()

        from iminuit import Minuit

        self.Minuit = Minuit

        if len(self.band_wave_cm) == 0:
            raise ValueError("At least one band must be specified.")

        self.bol_params_idx = np.array([P.reference_time, P.amplitude, P.rise_time, P.fall_time])
        self.temp_params_idx = np.array([P.reference_time, P.Tmin, P.delta_T, P.k_sig])

        band_to_index = {band: i for i, band in enumerate(self.band_wave_cm)}
        self.lookup_band_idx = np.vectorize(band_to_index.get)

        self.wave_cm_array = np.array(list(self.band_wave_cm.values()))

        # We are going to use it for the normalization of amplitude,
        # it is normally should be by the order of (\sigma_SB T^4) / (\pi B_\nu)
        self.average_nu = speed_of_light / np.mean(self.wave_cm_array)

    @classmethod
    def from_nm(cls, band_wave_nm, with_baseline=True):
        """Initialize from a dictionary of band names and their effective wavelengths in nm."""
        band_wave_nm = {band: 1e-7 * wavelength for band, wavelength in band_wave_nm.items()}
        return cls(band_wave_cm=band_wave_nm, with_baseline=with_baseline)

    @classmethod
    def from_angstrom(cls, band_wave_aa, with_baseline=True):
        """Initialize from a dictionary of band names and their effective wavelengths in angstroms."""
        band_wave_aa = {band: 1e-8 * wavelength for band, wavelength in band_wave_aa.items()}
        return cls(band_wave_cm=band_wave_aa, with_baseline=with_baseline)

    @staticmethod
    def _check_iminuit():
        if LeastSquares is None:
            raise ImportError(IMINUIT_IMPORT_ERROR)

        try:
            try:
                from packaging.version import parse as parse_version
            except ImportError:
                from distutils.version import LooseVersion as parse_version

            from iminuit import __version__
        except ImportError:
            raise ImportError(IMINUIT_IMPORT_ERROR)

        if parse_version(__version__) < parse_version("2.21.0"):
            raise ImportError(IMINUIT_IMPORT_ERROR)

    def bol_func(self, t, params):
        t0, amplitude, rise_time, fall_time = params[self.bol_params_idx]
        dt = t - t0
        return amplitude * np.exp(-dt / fall_time) / (1.0 + np.exp(-dt / rise_time))

    def temp_func(self, t, params):
        t0, T_min, delta_T, k_sig = params[self.temp_params_idx]
        dt = t - t0
        return T_min + delta_T / (1.0 + np.exp(dt / k_sig))

    @staticmethod
    def planck_nu(wave_cm, T):
        """Planck function in frequency units."""
        nu = speed_of_light / wave_cm
        return (
            (2 * planck_constant / speed_of_light**2)
            * nu**3
            / np.expm1(planck_constant * nu / (boltzman_constant * T))
        )

    def _lsq_model(self, x, *params):
        """Model function for the fit."""
        t, band_idx, wave_cm = x
        params = np.array(list(params))

        # Internally we use amplitude of F_bol / <nu> instead of F_bol.
        # It makes amplitude to be in the same units and the same order as
        # the baselines and input fluxes.
        params[P.amplitude] *= self.average_nu

        bol = self.bol_func(t, params)
        T = self.temp_func(t, params)
        flux = np.pi * self.planck_nu(wave_cm, T) / (sigma_sb * T**4) * bol
        if self.with_baseline:
            baseline_param_idx = band_idx + len(P)
            baselines = params[baseline_param_idx]
            flux += baselines
        return flux

    def model(self, t, band, *params):
        """Model function for the fit."""
        band_idx = self.lookup_band_idx(band)
        wave_cm = self.wave_cm_array[band_idx]
        params = np.array(list(params))
        # Internally we use amplitude of F_bol / <nu> instead of F_bol.
        params[P.amplitude] /= self.average_nu
        return self._lsq_model((t, band_idx, wave_cm), *params)

    @property
    def _baseline_names(self):
        if not self.with_baseline:
            return []
        return [f"baseline_{band}" for band in self.band_wave_cm]

    @property
    def names(self):
        """Names of the parameters."""
        return list(P.__members__) + self._baseline_names

    def _eval(self, *, t, m, sigma, band):
        # normalize input data
        t_shift = np.mean(t)
        t_scale = np.std(t)
        t = (t - t_shift) / t_scale

        # No-baseline assumes that zero flux corresponds to zero luminosity (intrinsic flux),
        # so no need to subtract the mean.
        m_shift_dict = dict.fromkeys(self.band_wave_cm, 0.0)
        if self.with_baseline:
            m_shift_dict = {b: np.mean(m[band == b]) for b in self.band_wave_cm}
        m_shift_array = np.array([m_shift_dict[band] for band in band])
        # copy array here to avoid modifying the input
        m = m - m_shift_array
        m_scale = np.std(m)
        # modify our own array in-place
        m /= m_scale

        sigma = sigma / m_scale

        t_amplitude = t[-1] - t[0]
        m_amplitude = np.ptp(m)

        initial_guesses = {
            "reference_time": 0.0,
            "amplitude": 1.0,
            "rise_time": 1.0,
            "fall_time": 1.0,
            "Tmin": 4000.0,
            "delta_T": 7000.0,
            "k_sig": 1.0,
        }
        limits = {
            "reference_time": (t[0] - 10 * t_amplitude, t[-1] + 10 * t_amplitude),
            "amplitude": (0.0, 10 * m_amplitude),
            "rise_time": (0.0, 10 * t_amplitude),
            "fall_time": (0.0, 10 * t_amplitude),
            "Tmin": (1e2, 1e6),  # K
            "delta_T": (0.0, 1e6),  # K
            "k_sig": (0.0, 10 * t_amplitude),
        }
        if self.with_baseline:
            initial_guesses.update(dict.fromkeys(self._baseline_names, 0.0))
            baseline_limits = (np.min(m) - 10 * m_amplitude, np.max(m))
            limits.update(dict.fromkeys(self._baseline_names, baseline_limits))

        band_idx = self.lookup_band_idx(band)
        wave_cm = self.wave_cm_array[band_idx]

        least_squares = LeastSquares(
            model=self._lsq_model,
            parameters=limits,
            x=(t, band_idx, wave_cm),
            y=m,
            yerror=sigma,
        )
        minuit = self.Minuit(least_squares, **initial_guesses)
        minuit.migrad()

        reduced_chi2 = minuit.fval / (len(t) - len(minuit.values))
        t0, amplitude, rise_time, fall_time = minuit.values[self.bol_params_idx]
        t0 = t0 * t_scale + t_shift
        # Internally we use amplitude of F_bol / <nu> instead of F_bol.
        amplitude = amplitude * m_scale * self.average_nu
        rise_time = rise_time * t_scale
        fall_time = fall_time * t_scale
        _t0, T_min, delta_T, k_sig = minuit.values[self.temp_params_idx]
        k_sig = k_sig * t_scale
        baselines = []
        if self.with_baseline:
            baselines = np.asarray(minuit.values[len(P) :]) * m_scale + np.array(list(m_shift_dict.values()))

        return np.r_[[t0, amplitude, rise_time, fall_time, T_min, delta_T, k_sig], baselines, reduced_chi2]

    # This is abstract class, but we could use default implementation while _eval is defined
    def _eval_and_fill(self, *, t, m, sigma, band, fill_value):
        return super()._eval_and_fill(t=t, m=m, sigma=sigma, band=band, fill_value=fill_value)
