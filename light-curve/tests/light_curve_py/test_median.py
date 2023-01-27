import numpy as np
from numpy.testing import assert_allclose

from light_curve.light_curve_py import Median


def test_median_1():
    n = 9
    m = np.arange(n)
    feature = Median()
    actual = feature(m, m, None)
    m = sorted(m)
    desired = m[n // 2]
    assert_allclose(actual, desired)


def test_median_2():
    n = 10
    m = np.arange(n)
    feature = Median()
    actual = feature(m, m, None)
    m = sorted(m)
    desired = (m[n // 2 - 1] + m[n // 2]) / 2
    assert_allclose(actual, desired)
