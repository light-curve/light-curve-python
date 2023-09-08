import numpy as np
from numpy.testing import assert_allclose

from light_curve.light_curve_py import Kurtosis


def test_kurtosis():
    m = [
        1.52403507,
        -0.47585435,
        1.30247288,
        -0.26769695,
        -1.89584998,
        0.30886293,
        -1.08824993,
        1.17991399,
        -0.61845487,
        0.12171829,
    ]
    feature = Kurtosis()
    desired = -0.704411
    actual = feature(np.linspace(0, 1, len(m)), m, None)
    assert_allclose(actual, desired)
