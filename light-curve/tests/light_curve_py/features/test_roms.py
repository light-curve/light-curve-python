import numpy as np
from numpy.testing import assert_allclose

from light_curve.light_curve_py import Roms


def test_roms_1():
    feature = Roms()
    n = 100
    m = np.ones(n)
    sigma = np.ones(n)
    actual = feature(m, m, sigma)
    desired = 0.0
    assert_allclose(actual, desired)


def test_roms_2():
    feature = Roms()
    n = 100
    x = np.linspace(0, 10, n)
    m = 10 * np.sin(x)
    sigma = np.ones(n)
    actual = feature(m, m, sigma)
    desired = sum(abs(m - np.median(m))) / (n - 1)
    assert_allclose(actual, desired)
