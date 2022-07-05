import numpy as np
from light_curve.light_curve_py import Cusum
from numpy.testing import assert_allclose


def test_cusum():
    m = [1, 2, 3, 4, 5, 5]
    feature = Cusum()
    actual = feature(np.linspace(0, 1, len(m)), m, None)
    desired = 0.408248290463863
    assert_allclose(actual, desired)
