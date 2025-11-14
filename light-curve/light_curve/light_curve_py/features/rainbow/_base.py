from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from light_curve.light_curve_py.dataclass_field import dataclass_field
from light_curve.light_curve_py.features._base import BaseMultiBandFeature
from light_curve.light_curve_py.features.rainbow._bands import Bands
from light_curve.light_curve_py.features.rainbow._parameters import create_parameters_class
from light_curve.light_curve_py.features.rainbow._scaler import MultiBandScaler, Scaler
from light_curve.light_curve_py.minuit_lsq import LeastSquares
from light_curve.light_curve_py.minuit_ml import MaximumLikelihood

__all__ = ["BaseRainbowFit"]


# CODATA 2018, grab from astropy
planck_constant = 6.62607004e-27  # erg s
speed_of_light = 2.99792458e10  # cm/s
boltzman_constant = 1.380649e-16  # erg/K
sigma_sb = 5.6703744191844314e-05  # erg/(cm^2 s K^4)


IMINUIT_IMPORT_ERROR = (
    "The `iminuit` package v2.21.0 or larger (exists for Python >= 3.8 only) is required for RainbowFit, "
    "please install it manually or reinstall light-curve package with [full] extra"
)


@dataclass()
class BaseRainbowFit(BaseMultiBandFeature):
    band_wave_cm: Dict[str, float]
    """Mapping of band names and their effective wavelengths in cm."""

    with_baseline: bool = dataclass_field(default=True, kw_only=True)
    """Whether to include a baseline in the fit, one per band."""

    fail_on_divergence: bool = dataclass_field(default=True, kw_only=True)
    """Fail (or return [fill_value]*n) if optimization hasn't converged"""

    @property
    def is_band_required(self) -> bool:
        return True

    @property
    def is_multiband_supported(self) -> bool:
        return True

    @property
    def size(self) -> int:
        return len(self.p)

    @staticmethod
    @abstractmethod
    def _common_parameter_names() -> List[str]:
        """Common parameter names."""
        return NotImplementedError

    @staticmethod
    @abstractmethod
    def _bolometric_parameter_names() -> List[str]:
        """Bolometric parameter names."""
        return NotImplementedError

    @staticmethod
    @abstractmethod
    def _temperature_parameter_names() -> List[str]:
        """Temperature parameter names."""
        return NotImplementedError

    def __post_init__(self) -> None:
        super().__post_init__()

        self._initialize_minuit()

        self.bands = Bands.from_dict(self.band_wave_cm)

        self.p = create_parameters_class(
            f"{self.__class__.__name__}Parameters",
            common=self._common_parameter_names(),
            bol=self._bolometric_parameter_names(),
            temp=self._temperature_parameter_names(),
            bands=self.bands,
            with_baseline=self.with_baseline,
        )

        if len(self.band_wave_cm) == 0:
            raise ValueError("At least one band must be specified.")

        # We are going to use it for the normalization of amplitude,
        # it is normally should be by the order of (\sigma_SB T^4) / (\pi B_\nu)
        self.average_nu = speed_of_light / self.bands.mean_wave_cm

        if self.with_baseline:
            self._lsq_model = self._lsq_model_with_baseline
        else:
            self._lsq_model = self._lsq_model_no_baseline

    def _initialize_minuit(self) -> None:
        self._check_iminuit()

        from iminuit import Minuit

        self.Minuit = Minuit

    @classmethod
    def from_nm(cls, band_wave_nm, **kwargs):
        """Initialize from a dictionary of band names and their effective wavelengths in nm."""
        band_wave_nm = {band: 1e-7 * wavelength for band, wavelength in band_wave_nm.items()}
        return cls(band_wave_cm=band_wave_nm, **kwargs)

    @classmethod
    def from_angstrom(cls, band_wave_aa, **kwargs):
        """Initialize from a dictionary of band names and their effective wavelengths in angstroms."""
        band_wave_aa = {band: 1e-8 * wavelength for band, wavelength in band_wave_aa.items()}
        return cls(band_wave_cm=band_wave_aa, **kwargs)

    @staticmethod
    def _check_iminuit():
        if LeastSquares is None:
            raise ImportError(IMINUIT_IMPORT_ERROR)

        if MaximumLikelihood is None:
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

    @abstractmethod
    def bol_func(self, t, params):
        """Bolometric light curve function."""
        raise NotImplementedError

    @abstractmethod
    def temp_func(self, t, params):
        """Temperature evolution function."""
        return NotImplementedError

    def _parameter_scalings(self) -> Dict[str, str]:
        """Rules for scaling/unscaling the parameters"""
        rules = {}

        if self.with_baseline:
            for band_name in self.bands.names:
                baseline_name = self.p.baseline_parameter_name(band_name)
                rules[baseline_name] = "baseline"

        return rules

    def _parameter_scale(self, name: str, t_scaler: Scaler, m_scaler: MultiBandScaler) -> float:
        """Return the scale factor to be applied to the parameter to unscale it"""
        scaling = self._parameter_scalings().get(name)
        if scaling == "time" or scaling == "timescale":
            return t_scaler.scale
        elif scaling == "flux":
            return m_scaler.scale

        return 1

    def _unscale_parameters(self, params, t_scaler: Scaler, m_scaler: MultiBandScaler) -> None:
        """Unscale parameters from internal units, in-place."""
        for name, scaling in self._parameter_scalings().items():
            if scaling == "time":
                params[self.p[name]] = t_scaler.undo_shift_scale(params[self.p[name]])

            elif scaling == "timescale":
                params[self.p[name]] = t_scaler.undo_scale(params[self.p[name]])

            elif scaling == "flux":
                params[self.p[name]] = m_scaler.undo_scale(params[self.p[name]])

            elif scaling == "baseline":
                band_name = self.p.baseline_band_name(name)
                baseline = params[self.p[name]]
                params[self.p[name]] = m_scaler.undo_shift_scale_band(baseline, band_name)

                pass

            elif scaling is None or scaling.lower() == "none":
                pass

            else:
                raise ValueError("Unsupported parameter scaling: " + scaling)

    def _unscale_errors(self, errors, t_scaler: Scaler, m_scaler: MultiBandScaler) -> None:
        """Unscale parameter errors from internal units, in-place."""
        for name in self.names:
            scale = self._parameter_scale(name, t_scaler, m_scaler)
            errors[self.p[name]] *= scale

    def _unscale_covariance(self, cov, t_scaler: Scaler, m_scaler: MultiBandScaler) -> None:
        """Unscale parameter covariance from internal units, in-place."""
        for name in self.names:
            scale = self._parameter_scale(name, t_scaler, m_scaler)
            i = self.p[name]
            cov[:, i] *= scale
            cov[i, :] *= scale

    @staticmethod
    def planck_nu(wave_cm, T):
        """Planck function in frequency units."""
        nu = speed_of_light / wave_cm
        return (
            (2 * planck_constant / speed_of_light**2) * nu**3 / np.expm1(planck_constant * nu / (boltzman_constant * T))
        )

    def _lsq_model_no_baseline(self, x, *params):
        """Model function for the fit."""
        t, _band_idx, wave_cm = x
        params = np.array(params)

        bol = self.bol_func(t, params)
        temp = self.temp_func(t, params)

        # Normalize the Planck function so that the result is of order unity
        norm = (sigma_sb * temp**4) / np.pi / self.average_nu  # Original "bolometric" normalization
        # peak_nu =  2.821 * boltzman_constant * temp / planck_constant # Wien displacement law
        # norm = self.planck_nu(speed_of_light / peak_nu, temp) # Peak = 1 normalization

        planck = self.planck_nu(wave_cm, temp) / norm

        flux = planck * bol

        return flux

    def _lsq_model_with_baseline(self, x, *params):
        flux = self._lsq_model_no_baseline(x, *params)

        _t, band_idx, _wave_cm = x
        params = np.array(params)
        baselines = params[self.p.lookup_baseline_idx_with_band_idx(band_idx)]
        flux += baselines

        return flux

    def model(self, t, band, *params):
        """Model function for the fit."""
        band_idx = self.bands.get_index(band)
        wave_cm = self.bands.index_to_wave_cm(band_idx)
        params = np.array(params)
        return self._lsq_model((t, band_idx, wave_cm), *params)

    @property
    def names(self):
        """Names of the parameters."""
        return list(self.p.__members__)

    @abstractmethod
    def _initial_guesses(self, t, m, sigma, band) -> Dict[str, float]:
        """Initial guesses for the fit parameters.

        t and m are *scaled* arrays. No baseline parameters are included.
        """
        return NotImplementedError

    def _baseline_initial_guesses(self, t, m, sigma, band) -> Dict[str, float]:
        """Initial guesses for the baseline parameters."""
        del t
        return {
            self.p.baseline_parameter_name(b): (np.median(m[band == b]) if np.sum(band == b) else 0)
            for b in self.bands.names
        }

    @abstractmethod
    def _limits(self, t, m, sigma, band) -> Dict[str, Tuple[float, float]]:
        """Limits for the fit parameters.

        t and m are *scaled* arrays. No baseline parameters are included.
        """
        return NotImplementedError

    def _baseline_limits(self, t, m, sigma, band) -> Dict[str, Tuple[float, float]]:
        """Limits for the baseline parameters."""
        del t
        limits = {}
        for b in self.bands.names:
            m_band = m[band == b]
            if len(m_band) > 0:
                lower = np.min(m_band) - 10 * np.ptp(m_band)
                upper = np.max(m_band)
            else:
                lower = 0
                upper = 0
            limits[self.p.baseline_parameter_name(b)] = (lower, upper)
        return limits

    def _eval(self, *, t, m, sigma, band):
        params, errors = self._eval_and_get_errors(t=t, m=m, sigma=sigma, band=band)
        return params

    # This is abstractmethod, but we could use default implementation while _eval is defined
    def _eval_and_fill(self, *, t, m, sigma, band, fill_value):
        return super()._eval_and_fill(t=t, m=m, sigma=sigma, band=band, fill_value=fill_value)

    def _eval_and_get_errors(
        self,
        *,
        t,
        m,
        sigma,
        band,
        upper_mask=None,
        get_initial=False,
        return_covariance=False,
        print_level=None,
        debug=False,
    ):
        # Initialize data scalers
        t_scaler = Scaler.from_time(t)
        m_scaler = MultiBandScaler.from_flux(m, band, with_baseline=self.with_baseline)

        # normalize input data
        t = t_scaler.do_shift_scale(t)
        m = m_scaler.do_shift_scale(m)
        sigma = m_scaler.do_scale(sigma)

        band_idx = self.bands.get_index(band)
        wave_cm = self.bands.index_to_wave_cm(band_idx)

        if self.with_baseline:
            initial_baselines = self._baseline_initial_guesses(t, m, sigma, band)
            m_corr = m - np.array([initial_baselines[self.p.baseline_parameter_name(b)] for b in band])

            # Compute initial guesses for the parameters on baseline-subtracted data
            initial_guesses = self._initial_guesses(t, m_corr, sigma, band)
            limits = self._limits(t, m_corr, sigma, band)

            initial_guesses.update(initial_baselines)
            limits.update(self._baseline_limits(t, m, sigma, band))
        else:
            # Compute initial guesses for the parameters on original data
            initial_guesses = self._initial_guesses(t, m, sigma, band)
            limits = self._limits(t, m, sigma, band)

        # least_squares = LeastSquares(
        cost_function = MaximumLikelihood(
            model=self._lsq_model,
            parameters=limits,
            x=(t, band_idx, wave_cm),
            y=m,
            yerror=sigma,
            upper_mask=upper_mask,
        )
        minuit = self.Minuit(cost_function, name=self.names, **initial_guesses)
        # TODO: expose these parameters through function arguments
        if print_level is not None:
            minuit.print_level = print_level
        minuit.strategy = 0  # We will need to manually call .hesse() on convergence anyway

        # Supposedly it is not the same as just setting iterate=10?..
        for i in range(10):
            minuit.migrad()

            if minuit.valid:
                minuit.hesse()
                # hesse() may may drive it invalid
                if minuit.valid:
                    break
            else:
                # That's what iterate is supposed to do?..
                minuit.simplex()
                # FIXME: it may drive the fit valid, but we will not have Hesse run on last iteration

        if debug:
            # Expose everything we have to outside, unscaled, for easier debugging
            self.minuit = minuit
            self.mparams = {
                "t": t,
                "band_idx": band_idx,
                "wave_cm": wave_cm,
                "m": m,
                "sigma": sigma,
                "limits": limits,
                "upper_mask": upper_mask,
                "initial_guesses": initial_guesses,
                "values": minuit.values,
                "errors": minuit.errors,
                "covariance": minuit.covariance,
            }

        if not minuit.valid and self.fail_on_divergence and not get_initial:
            raise RuntimeError("Fitting failed")

        reduced_chi2 = minuit.fval / (len(t) - self.size)

        if get_initial:
            # Reset the fitter so that it returns initial values instead of final ones
            minuit.reset()

        params = np.array(minuit.values)
        errors = np.array(minuit.errors)

        self._unscale_parameters(params, t_scaler, m_scaler)

        # Unscale errors
        self._unscale_errors(errors, t_scaler, m_scaler)

        return_values = np.r_[params, reduced_chi2], errors

        if return_covariance:
            # Unscale covaiance
            cov = np.array(minuit.covariance)
            self._unscale_covariance(cov, t_scaler, m_scaler)
            return_values += (cov,)

        return return_values

    def fit_and_get_errors(self, t, m, sigma, band, *, sorted=None, check=True, **kwargs):
        t, m, sigma, band = self._normalize_input(t=t, m=m, sigma=sigma, band=band, sorted=sorted, check=check)

        return self._eval_and_get_errors(t=t, m=m, sigma=sigma, band=band, **kwargs)
