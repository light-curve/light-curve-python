import numpy as np
from numpy.testing import assert_allclose

from light_curve.light_curve_py import MedianAbsoluteDeviation


def test_medabsdev():
    m = [1.0, 1.0, 3.0, 4.0, 7.0]
    feature = MedianAbsoluteDeviation()
    actual = feature(np.linspace(0, 1, len(m)), m, None)
    desired = 2.0
    assert_allclose(actual, desired)
