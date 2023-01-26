import numpy as np
from light_curve.light_curve_py import StandardDeviation
from numpy.testing import assert_allclose


def test_stdev():
    m = np.arange(10)
    feature = StandardDeviation()
    actual = feature(m, m, None)
    m_sum = sum((m - np.mean(m)) ** 2)
    desired = np.sqrt(m_sum / (len(m) - 1))
    assert_allclose(actual, desired)
