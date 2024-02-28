import numpy as np
from numpy.testing import assert_allclose

from light_curve.light_curve_py import Roms


def test_roms_const_data():
    feature = Roms()
    n = 100
    t = np.arange(n)
    m = np.ones_like(t)
    sigma = np.ones_like(t)
    actual = feature(t, m, sigma)
    desired = 0.0
    assert_allclose(actual, desired)


def test_roms_periodic_data():
    feature = Roms()
    n = 100
    t = np.linspace(0, 2 * np.pi, n)
    m = 2 * np.sin(t)
    sigma = np.ones_like(t)
    actual = feature(t, m, sigma)
    desired = 4 / np.pi
    assert_allclose(actual, desired, rtol=3 / np.sqrt(n))


def test_roms_norm_data_1():
    rng = np.random.default_rng(0)
    n = 100
    t = np.linspace(0, 1, n)
    m = rng.normal(0, 1, n)
    sigma = np.ones_like(t)
    feature = Roms()
    actual = feature(t, m, sigma)
    desired = 2 / np.sqrt((2 * np.pi))
    assert_allclose(actual, desired, rtol=3 / np.sqrt(n))


def test_roms_norm_data_2():
    rng = np.random.default_rng(0)
    n = 10000
    t = np.linspace(0, 1, n)
    m = rng.normal(0, 1, n)
    sigma = np.ones_like(t)
    feature = Roms()
    actual = feature(t, m, sigma)
    desired = 2 / np.sqrt((2 * np.pi))
    assert_allclose(actual, desired, rtol=3 / np.sqrt(n))


def test_roms_expon_data_1():
    rng = np.random.default_rng(0)
    n = 100
    t = np.linspace(0, 1, n)
    m = rng.exponential(2, n)
    sigma = np.ones_like(t)
    feature = Roms()
    actual = feature(t, m, sigma)
    desired = 2 * np.log(2)
    assert_allclose(actual, desired, rtol=3 / np.sqrt(n))


def test_roms_expon_data_2():
    rng = np.random.default_rng(0)
    n = 10000
    t = np.linspace(0, 1, n)
    m = rng.exponential(2, n)
    sigma = np.ones_like(t)
    feature = Roms()
    actual = feature(t, m, sigma)
    desired = 2 * np.log(2)
    assert_allclose(actual, desired, rtol=3 / np.sqrt(n))


def test_roms_gamma_data_1():
    rng = np.random.default_rng(0)
    n = 100
    t = np.linspace(0, 1, n)
    m = rng.gamma(2, 1, n)
    sigma = np.ones_like(t)
    feature = Roms()
    actual = feature(t, m, sigma)
    desired = 1.0518265193
    assert_allclose(actual, desired, rtol=3 / np.sqrt(n))


def test_roms_gamma_data_2():
    rng = np.random.default_rng(0)
    n = 10000
    t = np.linspace(0, 1, n)
    m = rng.gamma(2, 1, n)
    sigma = np.ones_like(t)
    feature = Roms()
    actual = feature(t, m, sigma)
    desired = 1.0518265193
    assert_allclose(actual, desired, rtol=3 / np.sqrt(n))
