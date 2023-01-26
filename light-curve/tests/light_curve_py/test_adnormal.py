import numpy as np
from light_curve.light_curve_py import AndersonDarlingNormal
from numpy.testing import assert_allclose


def test_adnormal():
    m = np.arange(0, 9)
    feature = AndersonDarlingNormal()
    desired = feature(m, m, None)
    actual = 0.155339690
    assert_allclose(actual, desired)
