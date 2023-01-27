import numpy as np
from numpy.testing import assert_allclose

from light_curve.light_curve_py import MaximumSlope


def test_maxslope():
    m = [2, 4, 15, 3]
    t = [1, 3, 4, 8]
    feature = MaximumSlope()
    actual = feature(t, m, None)
    desired = 11.0
    print(actual)
    assert_allclose(actual, desired)
