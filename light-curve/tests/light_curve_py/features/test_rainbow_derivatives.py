"""Regression guards for the analytic derivatives used by RainbowFit's fast path.

Every term exposes analytic derivatives (``derivatives`` for bolometric / temperature /
spectral terms, plus ``dvalue_dT`` for spectral terms), and ``BaseRainbowFit`` assembles
them into the full-model Jacobian ``_lsq_jac`` that feeds the iminuit analytic gradient and
the scipy ``least_squares`` backend. These tests pin every one of those derivatives against
central finite differences so a future change cannot silently break them.

Two layers:
- per-term checks (``test_*_term_derivatives``) localize a regression to a single term;
- ``test_full_model_jacobian`` checks the assembled ``_lsq_jac`` (chain rule, the ``T``
  shared between the temperature and blanketed spectral terms, and the baseline columns)
  for every bolometric x temperature x spectral combination, with and without baselines.
"""

import itertools

import numpy as np
import pytest

from light_curve.light_curve_py import RainbowFit
from light_curve.light_curve_py.features.rainbow.bolometric import bolometric_terms
from light_curve.light_curve_py.features.rainbow.spectral import spectral_terms
from light_curve.light_curve_py.features.rainbow.temperature import temperature_terms

BAND_WAVE_AA = {"g": 4770.0, "r": 6231.0, "i": 7625.0, "z": 9134.0}
WAVE_CM = np.array(list(BAND_WAVE_AA.values())) * 1e-8

# Representative, in-domain value for every parameter name that appears in any term.
# Chosen so each term sits in its smooth interior (away from clamping/clipping branches).
PARAM_VALUES = {
    "reference_time": 7.0,
    "amplitude": 1.5,
    "rise_time": 5.0,
    "fall_time": 20.0,
    "time1": 6.0,
    "time2": 12.0,
    "p": 0.3,
    "T": 12_000.0,
    "T_ratio": 0.6,
    "t_color": 8.0,
    "t_delay": 3.0,
    "lambda_scale": 0.3,
    "beta": 0.7,
    "sp_a": 0.4,
    "sp_b": -0.25,
    "spec_k": 1.3,
}

# Times chosen so no sample lands on reference_time=7.0 (avoids the non-smooth corner of
# the linexp bolometric and the branch boundaries of the others).
T_GRID = np.linspace(-12.7, 27.3, 33)
# Instantaneous temperature for the spectral term-level checks, deliberately different from
# the anchor PARAM_VALUES["T"] so the blanketed split (Planck core vs. extinction anchor) is
# exercised independently.
T_INSTANT = 9_000.0


def _fd_columns(func, params, h_rel=1e-6):
    """Central-difference Jacobian of ``func(params) -> (n_obs,)``, shape (n_params, n_obs)."""
    params = np.asarray(params, dtype=float)
    cols = []
    for i in range(len(params)):
        h = h_rel * max(1.0, abs(params[i]))
        pp = params.copy()
        pp[i] += h
        pm = params.copy()
        pm[i] -= h
        cols.append((func(pp) - func(pm)) / (2.0 * h))
    return np.array(cols)


def _assert_jac_close(analytic, fd, tol=1e-5):
    """Per-parameter relative error, normalized by that parameter's gradient scale.

    Columns whose gradient is tiny relative to the largest are floored at 0.1% of the global
    scale so finite-difference round-off in near-zero rows is not amplified into a failure;
    a genuinely broken derivative still produces an O(1) relative error.
    """
    analytic = np.asarray(analytic, dtype=float)
    fd = np.asarray(fd, dtype=float)
    # Terms with no parameters (e.g. plain Planck) have an empty derivative — nothing to
    # compare. Return before the shape check, where the (0, n_obs) analytic and the (0,)
    # finite-difference array would otherwise spuriously differ.
    if analytic.size == 0:
        return
    assert analytic.shape == fd.shape, f"shape mismatch {analytic.shape} vs {fd.shape}"
    global_scale = np.abs(fd).max()
    denom = np.maximum(np.abs(fd).max(axis=1, keepdims=True), global_scale * 1e-3)
    denom = np.where(denom > 0, denom, 1.0)
    rel = np.abs(analytic - fd) / denom
    assert rel.max() < tol, f"max relative derivative error {rel.max():.2e} exceeds {tol:.0e}"


@pytest.mark.parametrize("name", list(bolometric_terms))
def test_bolometric_term_derivatives(name):
    term = bolometric_terms[name]
    params = [PARAM_VALUES[p] for p in term.parameter_names()]
    analytic = term.derivatives(T_GRID, *params)
    fd = _fd_columns(lambda p: term.value(T_GRID, *p), params)
    _assert_jac_close(analytic, fd)


@pytest.mark.parametrize("name", list(temperature_terms))
def test_temperature_term_derivatives(name):
    term = temperature_terms[name]
    params = [PARAM_VALUES[p] for p in term.parameter_names()]
    analytic = term.derivatives(T_GRID, *params)
    fd = _fd_columns(lambda p: term.value(T_GRID, *p), params)
    _assert_jac_close(analytic, fd)


@pytest.mark.parametrize("name", list(spectral_terms))
def test_spectral_term_derivatives(name):
    term = spectral_terms[name]
    spec_params = [PARAM_VALUES[p] for p in term.parameter_names()]

    # dvalue_dT: derivative w.r.t. the *instantaneous* temperature (spectral params fixed).
    dvalue_dT = term.dvalue_dT(WAVE_CM, T_INSTANT, *spec_params)
    fd_T = _fd_columns(lambda T: term.value(WAVE_CM, T[0], *spec_params), [T_INSTANT])[0]
    _assert_jac_close(dvalue_dT[None, :], fd_T[None, :])

    # derivatives: w.r.t. each spectral parameter (rows follow parameter_names() order).
    analytic = term.derivatives(WAVE_CM, T_INSTANT, *spec_params)
    fd = _fd_columns(lambda p: term.value(WAVE_CM, T_INSTANT, *p), spec_params)
    _assert_jac_close(analytic, fd)


_COMBOS = list(itertools.product(bolometric_terms, temperature_terms, spectral_terms))


@pytest.mark.parametrize("bolometric,temperature,spectral", _COMBOS)
def test_full_model_jacobian(bolometric, temperature, spectral):
    """`_lsq_jac` vs finite differences of `_lsq_model`, for every term combination."""
    band = np.array([list(BAND_WAVE_AA)[i % len(BAND_WAVE_AA)] for i in range(len(T_GRID))])

    for with_baseline in (False, True):
        feature = RainbowFit.from_angstrom(
            BAND_WAVE_AA,
            with_baseline=with_baseline,
            bolometric=bolometric,
            temperature=temperature,
            spectral=spectral,
        )
        # Every current term exposes analytic derivatives, so the fast path must be wired up.
        assert feature._lsq_jac is not None, f"no analytic Jacobian for {bolometric}/{temperature}/{spectral}"

        band_idx = feature.bands.get_index(band)
        wave_cm = feature.bands.index_to_wave_cm(band_idx)
        x = (T_GRID, band_idx, wave_cm)

        params = np.array(
            [0.2 if name.startswith("baseline_") else PARAM_VALUES[name] for name in feature.names],
            dtype=float,
        )

        analytic = feature._lsq_jac(x, *params)
        fd = _fd_columns(lambda p: feature._lsq_model(x, *p), params)
        _assert_jac_close(analytic, fd)
