import numpy as np
from numpy.testing import assert_allclose

from light_curve.light_curve_py import Roms


def test_roms_1():
    feature = Roms()
    n = 100
    t = np.arange(n)
    m = np.ones_like(n)
    sigma = np.ones_like(n)
    actual = feature(t, m, sigma)
    desired = 0.0
    assert_allclose(actual, desired)


def test_roms_2():
    feature = Roms()
    n = 100
    t = np.linspace(0, 2 * np.pi, n)
    m = 2 * np.sin(t)
    sigma = np.ones_like(n)
    actual = feature(t, m, sigma)
    desired = 1.162
    assert_allclose(actual, desired, rtol=3 / np.sqrt(n))


def test_roms_3():
    rng = np.random.default_rng(0)
    n = 100
    t = np.linspace(0, 1, n)
    m = rng.normal(0, 1, n)
    sigma = np.ones_like(t)
    feature = Roms()
    actual = feature(t, m, sigma)
    desired = 0.797
    assert_allclose(actual, desired, rtol=3 / np.sqrt(n))


def test_roms_4():
    rng = np.random.default_rng(0)
    n = 10000
    t = np.linspace(0, 1, n)
    m = rng.normal(0, 1, n)
    sigma = np.ones_like(t)
    feature = Roms()
    actual = feature(t, m, sigma)
    desired = 0.797
    assert_allclose(actual, desired, rtol=3 / np.sqrt(n))


def test_roms_5():
    rng = np.random.default_rng(0)
    n = 100
    t = np.linspace(0, 1, n)
    m = rng.exponential(2, n)
    sigma = np.ones_like(t)
    feature = Roms()
    actual = feature(t, m, sigma)
    desired = 1.386
    assert_allclose(actual, desired, rtol=3 / np.sqrt(n))


def test_roms_6():
    rng = np.random.default_rng(0)
    n = 10000
    t = np.linspace(0, 1, n)
    m = rng.exponential(2, n)
    sigma = np.ones_like(t)
    feature = Roms()
    actual = feature(t, m, sigma)
    desired = 1.386
    assert_allclose(actual, desired, rtol=3 / np.sqrt(n))


def test_roms_7():
    rng = np.random.default_rng(0)
    n = 100
    t = np.linspace(0, 1, n)
    m = rng.gamma(2, 1, n)
    sigma = np.ones_like(t)
    feature = Roms()
    actual = feature(t, m, sigma)
    desired = 1.052
    assert_allclose(actual, desired, rtol=3 / np.sqrt(n))


def test_roms_8():
    rng = np.random.default_rng(0)
    n = 10000
    t = np.linspace(0, 1, n)
    m = rng.gamma(2, 1, n)
    sigma = np.ones_like(t)
    feature = Roms()
    actual = feature(t, m, sigma)
    desired = 1.052
    assert_allclose(actual, desired, rtol=3 / np.sqrt(n))
