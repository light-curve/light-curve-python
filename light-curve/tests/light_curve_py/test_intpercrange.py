import numpy as np
from numpy.testing import assert_allclose

from light_curve.light_curve_py import InterPercentileRange


def test_intpercrange_1():
    m = np.arange(1.0, 7.0)
    feature = InterPercentileRange()
    actual = feature(m, m)
    desired = 3.0
    assert_allclose(actual, desired)


def test_intpercrange_2():
    m = np.arange(1.0, 10.0)
    feature = InterPercentileRange(0.5)
    actual = feature(m, m)
    desired = 0.0
    assert_allclose(actual, desired)
