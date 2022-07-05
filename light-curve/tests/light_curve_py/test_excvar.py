import numpy as np
from numpy.testing import assert_allclose

from light_curve.light_curve_py import ExcessVariance


def test_excvar():
    m = [1, 1, 2, 3, 4, 5]
    sigma = [0.5, 0.5, 0.5, 0.5, 0.5, 0.2]
    feature = ExcessVariance()
    actual = feature(np.linspace(0, 1, len(m)), m, sigma)
    desired = 0.344765625
    assert_allclose(actual, desired)
