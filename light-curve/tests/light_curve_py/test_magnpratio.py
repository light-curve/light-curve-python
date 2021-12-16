import numpy as np
from numpy.testing import assert_allclose

from light_curve.light_curve_py import MagnitudePercentageRatio


def test_magnpratio_1():
    m = np.arange(1.0, 11.0)
    feature = MagnitudePercentageRatio()
    feature2 = MagnitudePercentageRatio(0.4, 0.05)
    actual = feature(m, m, None)
    actual2 = feature2(m, m, None)
    desired = 0.222222222
    assert_allclose(actual, actual2, desired)


def test_magnpratio_2():
    m = np.arange(1.0, 11.0)
    feature = MagnitudePercentageRatio(0.5, 0.05)
    actual = feature(m, m, None)
    desired = 0.0
    assert_allclose(actual, desired)
