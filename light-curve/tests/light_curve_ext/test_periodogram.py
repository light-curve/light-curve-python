import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.signal import lombscargle

from light_curve.light_curve_ext import Periodogram


def test_vs_lombscargle():
    rng = np.random.default_rng(None)
    n = 100

    t = np.sort(rng.normal(0, 1, n))
    m = np.sin(12.3 * t) + 0.2 * rng.normal(0, 1, n)
    scipy_y = (m - m.mean()) / m.std(ddof=1)

    freq_grids = [
        # This one fails with scipy for ZeroDivisionError
        # np.linspace(0.0, 100.0, 257),  # zero-based, step=2**k+1
        np.linspace(1.0, 100.0, 100),  # linear
        np.geomspace(1.0, 100.0, 100),  # arbitrary
    ]
    for freqs in freq_grids:
        licu_freqs, licu_power = Periodogram(freqs=freqs, fast=False).freq_power(t, m)
        assert_allclose(licu_freqs, freqs)
        licu_power2 = Periodogram(freqs=freqs, fast=False).power(t, m)
        assert_allclose(licu_power2, licu_power)
        scipy_power = lombscargle(t, scipy_y, freqs=freqs, precenter=True, normalize=False)
        assert_allclose(scipy_power, licu_power)


@pytest.mark.parametrize(
    "grid_name, freqs",
    [("np.linspace", np.linspace(1.0, 100.0, 100_000)), ("np.geomspace", np.geomspace(1.0, 100.0, 100_000))],
)
def test_benchmark_periodogram_rust(benchmark, grid_name, freqs):
    benchmark.group = f"periodogram_freqs={grid_name}"
    benchmark.name = "rust"

    rng = np.random.default_rng(0)
    n = 100

    t = np.sort(rng.normal(0, 1, n))
    m = np.sin(12.3 * t) + 0.2 * rng.normal(0, 1, n)

    fe = Periodogram(freqs=freqs, fast=False)
    benchmark(fe.freq_power, t, m)


@pytest.mark.parametrize(
    "grid_name, freqs",
    [("np.linspace", np.linspace(1.0, 100.0, 100_000)), ("np.geomspace", np.geomspace(1.0, 100.0, 100_000))],
)
def test_benchmark_periodogram_scipy(benchmark, grid_name, freqs):
    benchmark.group = f"periodogram_freqs={grid_name}"
    benchmark.name = "scipy"

    rng = np.random.default_rng(0)
    n = 100

    t = np.sort(rng.normal(0, 1, n))
    m = np.sin(12.3 * t) + 0.2 * rng.normal(0, 1, n)
    y = (m - m.mean()) / m.std(ddof=1)

    benchmark(lombscargle, t, y, freqs, precenter=True, normalize=False)


def test_different_freq_grids():
    rng = np.random.default_rng(None)

    rng = np.random.default_rng(None)
    n = 100

    t = np.sort(rng.normal(0, 1, n))
    m = np.sin(12.3 * t) + 0.2 * rng.normal(0, 1, n)

    base_grid = np.r_[0:100:257j]
    base_power = None

    freq_grids = [
        base_grid,  # zero-based, step=2**k+1
        np.r_[base_grid, base_grid[-1] + base_grid[1]],  # linear
        np.r_[base_grid, 200.0],  # arbitrary
    ]
    for freqs in freq_grids:
        licu_freqs, licu_power = Periodogram(freqs=freqs, fast=False).freq_power(t, m)
        assert_allclose(licu_freqs, freqs)
        licu_power2 = Periodogram(freqs=freqs, fast=False).power(t, m)
        assert_allclose(licu_power2, licu_power)
        if base_power is None:
            base_power = licu_power
        else:
            assert_allclose(licu_power[:-1], base_power)


def test_failure_for_wrong_freq_grids():
    with pytest.raises(ValueError):
        # Too short
        Periodogram(freqs=[1.0], fast=False)
    with pytest.raises(ValueError):
        # Too short
        Periodogram(freqs=[1.0], fast=True)
    with pytest.raises(ValueError):
        # size is not 2**k + 1
        Periodogram(freqs=np.linspace(0.0, 100.0, 100), fast=True)
    with pytest.raises(ValueError):
        # Doesn't start with 0.0
        Periodogram(freqs=np.linspace(1.0, 100.0, 257), fast=True)


def test_normalization_default_is_psd():
    """Default normalization should match explicit 'psd'."""
    rng = np.random.default_rng(42)
    n = 100
    t = np.sort(rng.uniform(0, 10, n))
    m = np.sin(2.0 * t) + 0.1 * rng.normal(0, 1, n)
    freqs = np.linspace(1.0, 50.0, 200)

    power_default = Periodogram(freqs=freqs, fast=False).power(t, m)
    power_psd = Periodogram(freqs=freqs, fast=False, normalization="psd").power(t, m)
    assert_allclose(power_default, power_psd)


def test_normalization_invalid():
    """Invalid normalization string should raise ValueError."""
    with pytest.raises(ValueError, match="normalization must be one of"):
        Periodogram(normalization="invalid")


@pytest.mark.parametrize("normalization", ["standard", "model", "log"])
def test_normalization_vs_astropy(normalization):
    """Compare normalizations against astropy LombScargle.

    'psd' is excluded because our convention matches scipy.signal.lombscargle
    (normalize=False) on variance-normalized data, while astropy's 'psd'
    uses a different scaling convention.
    """
    astropy_ts = pytest.importorskip("astropy.timeseries")

    rng = np.random.default_rng(42)
    n = 100
    t = np.sort(rng.uniform(0, 10, n))
    m = np.sin(2.0 * t) + 0.1 * rng.normal(0, 1, n)
    freqs = np.linspace(1.0, 50.0, 200)
    # astropy uses ordinary frequency, not angular
    astropy_freq = freqs / (2.0 * np.pi)

    ls = astropy_ts.LombScargle(t, m, fit_mean=False, center_data=True)

    licu_power = Periodogram(freqs=freqs, fast=False, normalization=normalization).power(t, m)
    astropy_power = ls.power(astropy_freq, normalization=normalization)
    assert_allclose(licu_power, astropy_power, rtol=1e-5)
