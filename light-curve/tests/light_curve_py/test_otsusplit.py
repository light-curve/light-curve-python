import numpy as np
from numpy.testing import assert_allclose

from light_curve.light_curve_py import OtsuSplit


def test_otsu_1():
    feature = OtsuSplit()
    m = np.array([0.5, 1.5, 1.5, 1.5])
    t = np.linspace(0, 1.7, 4)
    actual = feature(t, m, None)
    desired = [1.0, 0.0, 0.0, 0.25]
    assert_allclose(actual, desired)


def test_otsu_2():
    feature = OtsuSplit()
    m = np.array([0.45, 0.5, 0.52])
    t = np.linspace(0, 1.7, 3)
    actual = feature(t, m, None)
    desired = [0.06, 0.0, np.std([0.5, 0.52], ddof=1), 0.33333]
    assert_allclose(actual, desired, rtol=1e-04)


def test_otsu_thr():
    feature = OtsuSplit()
    m = np.array([0.45, 0.5, 0.52])
    actual = feature.threshold(m)
    desired = 0.45
    assert_allclose(actual, desired, rtol=1e-04)
