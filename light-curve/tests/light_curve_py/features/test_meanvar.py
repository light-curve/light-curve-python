import numpy as np
from numpy.testing import assert_allclose

from light_curve.light_curve_py import MeanVariance


def test_meanvar():
    feature = MeanVariance()
    m = [1, 1, 2, 2, 3, 3, 4, 4]
    actual = feature(np.linspace(0, 1, len(m)), m, None)
    desired = 0.47809144373375745
    assert_allclose(actual, desired)
