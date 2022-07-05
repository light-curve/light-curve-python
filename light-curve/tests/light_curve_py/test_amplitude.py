import numpy as np
from light_curve.light_curve_py import Amplitude
from numpy.testing import assert_allclose


def test_amplitude():
    n = 11
    feature = Amplitude()
    m = np.arange(n)
    desired = (n - 1) / 2
    actual = feature(m, m, None)
    assert_allclose(actual, desired)
