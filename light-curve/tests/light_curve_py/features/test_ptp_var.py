import numpy as np
from numpy.testing import assert_allclose

from light_curve.light_curve_py import PeakToPeakVar


def test_ptpvar_const_data():
    feature = PeakToPeakVar()
    n = 100
    t = np.arange(n)
    m = np.ones_like(t)
    sigma = 0.1 * np.ones_like(t)
    actual = feature(t, m, sigma)
    desired = -0.1
    assert_allclose(actual, desired)


def test_ptpvar_periodic_data():
    feature = PeakToPeakVar()
    n = 100
    t = np.linspace(0, np.pi, n)
    m = np.sin(t)
    sigma = 0.1 * np.ones_like(t)
    actual = feature(t, m, sigma)
    desired = 0.8
    assert_allclose(actual, desired, rtol=3 / np.sqrt(n))


def test_ptpvar_norm_data_1():
    rng = np.random.default_rng(0)
    n = 100
    t = np.linspace(0, 1, n)
    m = np.abs(rng.normal(0, 1, n))
    sigma = 0.2 * np.ones_like(t)
    feature = PeakToPeakVar()
    actual = feature(t, m, sigma)
    desired = 1
    assert_allclose(actual, desired, rtol=3 / np.sqrt(n))


def test_ptpvar_norm_data_2():
    rng = np.random.default_rng(0)
    n = 10000
    t = np.linspace(0, 1, n)
    m = np.abs(rng.normal(0, 1, n))
    sigma = 0.2 * np.ones_like(t)
    feature = PeakToPeakVar()
    actual = feature(t, m, sigma)
    desired = 1
    assert_allclose(actual, desired, rtol=3 / np.sqrt(n))


def test_ptpvar_expon_data_1():
    rng = np.random.default_rng(0)
    n = 100
    t = np.linspace(0, 1, n)
    m = rng.exponential(2, n)
    sigma = np.ones_like(t)
    feature = PeakToPeakVar()
    actual = feature(t, m, sigma)
    desired = 1
    assert_allclose(actual, desired, rtol=3 / np.sqrt(n))


def test_ptpvar_expon_data_2():
    rng = np.random.default_rng(0)
    n = 10000
    t = np.linspace(0, 1, n)
    m = rng.exponential(2, n)
    sigma = np.ones_like(t)
    feature = PeakToPeakVar()
    actual = feature(t, m, sigma)
    desired = 1
    assert_allclose(actual, desired, rtol=3 / np.sqrt(n))