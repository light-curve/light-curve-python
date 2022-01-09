import numpy as np
from numpy.testing import assert_allclose

from light_curve.light_curve_py import Mean


def test_mean_1():
    n = 10
    feature = Mean()
    m = np.arange(10)
    desired = sum(m) / n
    actual = feature(m, m, None)
    assert_allclose(actual, desired)


def test_mean_2():
    feature = Mean()
    m = np.linspace(0, 50, 100)
    desired = sum(m) / len(m)
    actual = feature(m, m, None)
    assert_allclose(actual, desired)
