import numpy as np
from numpy.testing import assert_allclose

from light_curve.light_curve_py import ReducedChi2


def test_redchi2_equal_sigma():
    m = np.array([1.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    sigma = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    feature = ReducedChi2()
    desired = feature(np.linspace(0, 1, len(m)), m, sigma)
    actual = 10.666667
    assert_allclose(actual, desired)


def test_redchi2_different_sigma():
    m = np.arange(6)
    sigma = np.array([0.5, 1.0, 0.5, 1.0, 0.5, 1.0])
    feature = ReducedChi2()
    desired = feature(np.linspace(0, 1, len(m)), m, sigma)
    actual = 8.48
    assert_allclose(actual, desired)
