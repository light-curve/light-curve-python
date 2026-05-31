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


sigma_sb = 5.6703744191844314e-05  # erg/(cm^2 s K^4)
speed_of_light = 2.99792458e10  # cm/s


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

    optimizer: str = dataclass_field(default="iminuit", kw_only=True)
    """Optimizer backend: 'iminuit' (default, robust Migrad) or 'least_squares'
    (scipy Trust Region Reflective). The latter is experimental and only applies to
    measurement-only fits with an analytic Jacobian; it transparently falls back to
    iminuit for upper-limit fits or term combinations without analytic derivatives."""

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
    def _common_bol_temp_parameter_names() -> List[str]:
        """Common parameter names beween bolometric and temperature models."""
        return NotImplementedError

    @staticmethod
    @abstractmethod
    def _common_temp_spec_parameter_names() -> List[str]:
        """Common parameter names beween temperature and spectral models."""
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

    @staticmethod
    @abstractmethod
    def _spectral_parameter_names() -> List[str]:
        """Spectral model parameter names."""
        return NotImplementedError

    def __post_init__(self) -> None:
        super().__post_init__()

        self._initialize_minuit()

        self.bands = Bands.from_dict(self.band_wave_cm)

        self.p = create_parameters_class(
            f"{self.__class__.__name__}Parameters",
            common_bol_temp=self._common_bol_temp_parameter_names(),
            common_temp_spec=self._common_temp_spec_parameter_names(),
            bol=self._bolometric_parameter_names(),
            temp=self._temperature_parameter_names(),
            spec=self._spectral_parameter_names(),
            bands=self.bands,
            with_baseline=self.with_baseline,
        )

        if len(self.band_wave_cm) == 0:
            raise ValueError("At least one band must be specified.")

        if self.optimizer not in ("iminuit", "least_squares"):
            raise ValueError(f"Unknown optimizer {self.optimizer!r}, expected 'iminuit' or 'least_squares'")

        # We are going to use it for the normalization of amplitude,
        # it is normally should be by the order of (\sigma_SB T^4) / (\pi B_\nu)
        self.average_nu = speed_of_light / self.bands.mean_wave_cm

        if self.with_baseline:
            self._lsq_model = self._lsq_model_with_baseline
        else:
            self._lsq_model = self._lsq_model_no_baseline

        # Analytic Jacobian is available only when every term provides the
        # right derivative method. Subclasses (e.g. RainbowFit) wire the
        # term-level checks via `_supports_analytic_jac`.
        if self._supports_analytic_jac():
            self._lsq_jac = self._lsq_jac_with_baseline if self.with_baseline else self._lsq_jac_no_baseline
        else:
            self._lsq_jac = None

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

        spectral = self.spectral_func(wave_cm, temp, params)

        SED = spectral / norm
        flux = SED * bol

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

    def _supports_analytic_jac(self) -> bool:
        """Whether all of the current terms expose the methods needed by `_lsq_jac_*`.

        Default is False; concrete subclasses (RainbowFit) override based on the
        term combination they actually compose.
        """
        return False

    def _lsq_jac_no_baseline(self, x, *params):
        """Jacobian d flux / d parameter, shape (n_params, n_obs)."""
        t, _band_idx, wave_cm = x
        params = np.array(params)
        n_params = len(params)

        spec_params = params[self.p.all_spec_idx]

        bol = self.bol_func(t, params)
        temp = self.temp_func(t, params)
        norm = (sigma_sb * temp**4) / np.pi / self.average_nu
        spectral = self.spectral_func(wave_cm, temp, params)
        g = spectral / norm  # spec / norm

        bol_jac = self.bolometric.derivatives(t, *params[self.p.all_bol_idx])
        temp_jac = self.temperature.derivatives(t, *params[self.p.all_temp_idx])
        dspec_dT = self.spectral.dvalue_dT(wave_cm, temp, *spec_params)

        # d(spec/norm)/dT = dspec/dT / norm  -  (spec/norm) · 4/T
        dg_dT = dspec_dT / norm - g * 4.0 / temp

        jac = np.zeros((n_params, len(t)))
        # Bolometric parameters (including shared t0): df/dθ = (dB/dθ) · g
        for i, idx in enumerate(self.p.all_bol_idx):
            jac[idx] += bol_jac[i] * g
        # Temperature parameters (including shared t0): df/dθ = bol · dg/dT · dT/dθ
        for i, idx in enumerate(self.p.all_temp_idx):
            jac[idx] += bol * dg_dT * temp_jac[i]
        # Spectral parameters: df/dθ_spec = bol · (∂spec/∂θ_spec) / norm
        if len(self.p.all_spec_idx) > 0:
            spec_jac = self.spectral.derivatives(wave_cm, temp, *spec_params)
            inv_norm = 1.0 / norm
            for i, idx in enumerate(self.p.all_spec_idx):
                jac[idx] += bol * spec_jac[i] * inv_norm

        return jac

    def _lsq_jac_with_baseline(self, x, *params):
        jac = self._lsq_jac_no_baseline(x, *params)
        _t, band_idx, _wave_cm = x
        # ∂flux/∂baseline_b = 1 where band == b, else 0.
        param_idx_per_obs = self.p.baseline_idx[band_idx]
        jac[param_idx_per_obs, np.arange(len(band_idx))] = 1.0
        return jac

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

        # Force all parameter dictionaries to follow enum / Minuit order
        initial_guesses = {name: initial_guesses[name] for name in self.names}
        limits = {name: limits[name] for name in self.names}

        # scipy least_squares is a least-squares-only optimizer: it needs the analytic
        # Jacobian and cannot represent the Tobit upper-limit likelihood. Fall back to
        # iminuit when either prerequisite is missing.
        if self.optimizer == "least_squares" and upper_mask is None and self._lsq_jac is not None:
            lsq_result = self._fit_least_squares(
                t_scaler,
                m_scaler,
                (t, band_idx, wave_cm),
                m,
                sigma,
                initial_guesses,
                limits,
                get_initial=get_initial,
                return_covariance=return_covariance,
                debug=debug,
            )
            # TRF returns None when it failed to converge — sparse, ill-conditioned light
            # curves where it crawls along degenerate directions and exhausts its budget
            # without reaching first-order optimality. iminuit's barrier + simplex-restart
            # loop handles exactly these, so we fall through to it for the rare (~1%) hard
            # case rather than returning a bad fit.
            if lsq_result is not None:
                return lsq_result

        # Analytic Jacobian is only safe without upper limits; the gradient code
        # in MaximumLikelihood explicitly refuses upper_mask + jac.
        jac = self._lsq_jac if (self._lsq_jac is not None and upper_mask is None) else None
        cost_function = MaximumLikelihood(
            model=self._lsq_model,
            parameters=limits,
            x=(t, band_idx, wave_cm),
            y=m,
            yerror=sigma,
            upper_mask=upper_mask,
            jac=jac,
        )
        # `grad=False` forces iminuit to fall back to its numerical gradient.
        # If we passed `grad=None`, iminuit would auto-detect `cost_function.grad`
        # and call it even when no Jacobian is wired in.
        minuit_grad = cost_function.grad if jac is not None else False
        minuit = self.Minuit(cost_function, name=self.names, grad=minuit_grad, **initial_guesses)
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

    @staticmethod
    def _lsq_covariance(jac_residual):
        """Covariance ``inv(JᵀJ)`` from the residual Jacobian at the solution.

        ``jac_residual`` is ``d residual / d params`` (shape ``(n_obs, n_params)``)
        where the residuals are already weighted by ``1/σ``, so ``JᵀJ`` is the
        Gauss-Newton Hessian of the 0.5·χ² objective and its inverse is the parameter
        covariance (same construction scipy's ``curve_fit`` uses with absolute sigmas).

        Falls back to the Moore-Penrose pseudo-inverse when ``JᵀJ`` is singular or
        numerically non-positive-definite. That degeneracy means the data does not
        constrain some parameter direction (e.g. the blanketed ``lambda_scale`` on a
        cool source where UV extinction is negligible) — an unidentified parameter, not
        a failed fit — so we still return a best-effort covariance rather than None.
        (Caveat: ``pinv`` reports a *zero* variance for a perfectly unconstrained
        direction; the converged parameters and χ² are unaffected, only those error
        bars are unreliable.) Always returns a finite, symmetric matrix.
        """
        jtj = jac_residual.T @ jac_residual
        try:
            cov = np.linalg.inv(jtj)
            if np.all(np.isfinite(cov)) and np.all(np.diag(cov) > 0):
                return cov
        except np.linalg.LinAlgError:
            pass
        return np.linalg.pinv(jtj)

    def _fit_least_squares(
        self,
        t_scaler,
        m_scaler,
        x,
        m,
        sigma,
        initial_guesses,
        limits,
        *,
        get_initial,
        return_covariance,
        debug,
    ):
        """Experimental scipy Trust Region Reflective backend for measurement-only fits.

        Reuses the same scaled data, bounds, analytic model and Jacobian as the iminuit
        path; only the optimizer and the covariance source differ. Hard box bounds
        replace iminuit's soft barrier penalty, and the covariance comes from the
        Gauss-Newton ``inv(JᵀJ)`` at the solution rather than from hesse().

        Returns the usual ``_eval_and_get_errors`` tuple, or ``None`` when TRF fails to
        converge so the caller can fall back to iminuit for that light curve.
        """
        from scipy.optimize import least_squares

        names = self.names
        lower = np.array([limits[name][0] for name in names], dtype=float)
        upper = np.array([limits[name][1] for name in names], dtype=float)
        p0 = np.array([initial_guesses[name] for name in names], dtype=float)
        span = upper - lower
        # TRF stalls when started exactly on a bound whose gradient is degenerate there
        # (e.g. the blanketed `lambda_scale` guess equals its lower bound, in the
        # no-extinction limit). Nudge only such on-bound parameters a little into the
        # interior — interior guesses, which are the vast majority, are left untouched
        # so wide-bound parameters like temperature are unaffected. Then enforce strict
        # feasibility for TRF.
        on_lower = p0 <= lower + 1e-8 * span
        on_upper = p0 >= upper - 1e-8 * span
        p0[on_lower] = lower[on_lower] + 0.05 * span[on_lower]
        p0[on_upper] = upper[on_upper] - 0.05 * span[on_upper]
        p0 = np.clip(p0, lower + 1e-10 * span, upper - 1e-10 * span)

        inv_sigma = 1.0 / sigma

        def residual(params):
            return (self._lsq_model(x, *params) - m) * inv_sigma

        def jacobian(params):
            return (self._lsq_jac(x, *params) * inv_sigma).T  # (n_obs, n_params)

        if get_initial:
            params = p0
            cov_scaled = None
            valid = True
            chi2 = float(np.sum(residual(p0) ** 2))
        else:
            # The default TRF budget (~100·n_params) is too tight for the ill-conditioned
            # blanketed Jacobian on some real light curves: they reach the optimum but hit
            # the iteration cap before TRF can *certify* convergence, leaving result.success
            # False. 300·n_params sits at the knee of the certify-rate/cost curve on real
            # LSST data (halves the spurious-invalid rate for a small time cost; larger caps
            # buy almost nothing more). It only bites the stuck tail, so a typical fit, which
            # converges in tens of evals, is unaffected.
            result = least_squares(
                residual,
                p0,
                jac=jacobian,
                bounds=(lower, upper),
                method="trf",
                x_scale="jac",
                max_nfev=300 * len(names),
            )
            if not result.success:
                # TRF could not certify convergence (sparse, ill-conditioned light
                # curves where it crawls along degenerate directions and exhausts its
                # budget). Signal the caller to fall back to iminuit rather than return a
                # possibly-bad fit. A singular covariance, by contrast, is NOT failure —
                # it only means an unidentified parameter direction — so it does not get
                # here; result.success is the sole convergence criterion.
                return None
            params = result.x
            chi2 = float(np.sum(result.fun**2))
            cov_scaled = self._lsq_covariance(result.jac)
            valid = True

        if cov_scaled is not None:
            # Clamp tiny negative diagonals from roundoff / pinv before the sqrt.
            errors = np.sqrt(np.clip(np.diag(cov_scaled), 0.0, None))
        else:
            errors = np.full(len(params), np.nan)

        if debug:
            # Mirror the iminuit debug payload as closely as the scipy path allows.
            self.minuit = None
            self.mparams = {
                "limits": limits,
                "initial_guesses": initial_guesses,
                "values": params,
                "errors": errors,
                "covariance": cov_scaled,
                "valid": valid,
            }

        # Match iminuit's convention: it reports MaximumLikelihood.fval / dof, and fval
        # is the 0.5·χ² negative-log-likelihood, hence the factor of one half here.
        reduced_chi2 = 0.5 * chi2 / (len(m) - self.size)

        params = params.copy()
        self._unscale_parameters(params, t_scaler, m_scaler)
        self._unscale_errors(errors, t_scaler, m_scaler)

        return_values = np.r_[params, reduced_chi2], errors

        if return_covariance:
            cov = cov_scaled.copy() if cov_scaled is not None else np.full((len(params), len(params)), np.nan)
            self._unscale_covariance(cov, t_scaler, m_scaler)
            return_values += (cov,)

        return return_values

    def fit_and_get_errors(self, t, m, sigma, band, *, sorted=None, check=True, **kwargs):
        t, m, sigma, band = self._normalize_input(t=t, m=m, sigma=sigma, band=band, sorted=sorted, check=check)

        return self._eval_and_get_errors(t=t, m=m, sigma=sigma, band=band, **kwargs)
