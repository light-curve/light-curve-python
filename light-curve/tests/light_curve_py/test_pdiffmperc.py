import numpy as np
from light_curve.light_curve_py import PercentDifferenceMagnitudePercentile
from numpy.testing import assert_allclose


def test_pdiffmperc():
    m = np.arange(1.0, 7.0)
    feature = PercentDifferenceMagnitudePercentile(0.25)
    actual = feature(m, m)
    desired = 3.0 / 3.5
    assert_allclose(actual, desired)
