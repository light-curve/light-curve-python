import numpy as np
import pytest

from light_curve.light_curve_py import LinearFit


def test_no_init_bands_no_input_bands():
    """Announce no bands, use no bands"""
    t = [1.0, 2.0, 3.0, 4.0, 5.0]
    m = [1.0, 2.0, 3.0, 4.0, 5.0]
    sigma = [1.0, 1.0, 1.0, 1.0, 1.0]
    feature = LinearFit(bands=None)
    _values = feature(t, m, sigma)


def test_init_bands_no_input_bands():
    """Announce bands, use no bands"""
    t = [1.0, 2.0, 3.0, 4.0, 5.0]
    m = [1.0, 2.0, 3.0, 4.0, 5.0]
    sigma = [1.0, 1.0, 1.0, 1.0, 1.0]
    feature = LinearFit(bands=["g"])
    with pytest.raises(ValueError):
        _values = feature(t, m, sigma)


def test_no_init_bands_input_bands():
    """Announce no bands, use input bands"""
    t = [1.0, 2.0, 3.0, 4.0, 5.0]
    m = [1.0, 2.0, 3.0, 4.0, 5.0]
    sigma = [1.0, 1.0, 1.0, 1.0, 1.0]
    bands = ["g", "g", "g", "r", "r"]
    feature = LinearFit(bands=None)
    with pytest.raises(ValueError):
        _values = feature(t, m, sigma, bands)


def test_init_bands_eq_input_bands():
    """Announce no bands, use input bands"""
    t = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    m = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    sigma = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    bands = ["g", "g", "g", "r", "r", "r"]
    feature = LinearFit(bands=["g", "r"])
    values = feature(t, m, sigma, bands)
    assert 2 == feature.n_bands
    assert values.size == feature.size
    assert values.size == feature.size_single_band * feature.n_bands


def test_init_bands_less_input_bands():
    t = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    m = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    sigma = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    bands = ["g", "g", "g", "r", "r", "r"]
    feature = LinearFit(bands=["g"])
    values = feature(t, m, sigma, bands)
    assert 1 == feature.n_bands
    assert values.size == feature.size
    assert values.size == feature.size_single_band


def test_init_bands_more_input_bands():
    t = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    m = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    sigma = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    bands = ["g"] * 6
    feature = LinearFit(bands=["g", "r"])
    # "r" band is virtually existing, but having zero observations
    with pytest.raises(ValueError):
        _values = feature(t, m, sigma, bands, fill_value=None)
    # ... so it should be filled with the fill_value
    values = feature(t, m, sigma, bands, fill_value=np.nan)
    assert 2 == feature.n_bands
    assert values.size == feature.size
    assert values.size == feature.size_single_band * feature.n_bands
    assert np.all(np.isfinite(values[: feature.size_single_band]))
    assert np.all(np.isnan(values[feature.size_single_band :]))


def test_init_bands_different_input_bands():
    t = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    m = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    sigma = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    bands = ["g"] * 6
    feature = LinearFit(bands=["r"])
    # "r" band is virtually existing, but having zero observations
    with pytest.raises(ValueError):
        _values = feature(t, m, sigma, bands, fill_value=None)
    # ... so it should be filled with the fill_value
    values = feature(t, m, sigma, bands, fill_value=-999.0)
    assert 1 == feature.n_bands
    assert values.size == feature.size == feature.size_single_band
    np.testing.assert_array_equal(values, -999.0)
