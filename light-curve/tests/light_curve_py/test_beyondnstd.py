from numpy.testing import assert_allclose

from light_curve.light_curve_py import BeyondNStd


def test_beyondnstd_1():
    m = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
    feature = BeyondNStd(1.0)
    feature_2 = BeyondNStd()
    actual = feature(m, m)
    actual_2 = feature_2(m, m)
    desired = 1 / 6
    assert_allclose(actual, desired, actual_2)


def test_beyondnsts_2():
    m = [1.0, 18.0, 45.0, 100.0]
    feature = BeyondNStd(3.0)
    actual = feature(m, m)
    desired = 0.0
    assert_allclose(actual, desired)
