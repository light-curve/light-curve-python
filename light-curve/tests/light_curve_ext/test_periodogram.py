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
        scipy_power = lombscargle(t, scipy_y, freqs=freqs, precenter=True, normalize=False)
        assert_allclose(scipy_power, licu_power)


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
