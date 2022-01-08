import numpy as np
from numpy.testing import assert_allclose

from light_curve.light_curve_py import Eta, EtaE


def test_etae_1():
    a = np.array([1, 2, 3, 8])
    t = np.array([1, 3, 5, 7])
    feature = EtaE()
    feature_2 = Eta()
    actual = feature(t, a, None)
    actual_2 = feature_2(t, a, None)
    assert_allclose(actual, actual_2)


def test_etae_2():
    a = np.array([1, 2, 3, 8, 11])
    t = np.array([1, 2, 7, 9, 10])
    feature = EtaE()
    desired = 1.114434
    actual = feature(t, a, None)
    assert_allclose(actual, desired, rtol=1e-06)
