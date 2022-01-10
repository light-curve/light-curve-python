import numpy as np
from numpy.testing import assert_allclose

from light_curve.light_curve_py import MedianBufferRangePercentage


def test_medbufrperc():
    m = np.arange(1.0, 8.0)
    feature = MedianBufferRangePercentage()
    actual = feature(m, m, None)
    desired = 1 / 7
    assert_allclose(actual, desired)
