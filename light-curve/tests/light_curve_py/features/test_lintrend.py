import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import curve_fit

from light_curve.light_curve_py import LinearTrend


def test_lintrend_1():
    def f(x, slope, c):
        return x * slope + c

    m = np.array([2.0, 4.0, 7.0, 10.0, 15.0, 28.0])
    t = np.array([1.0, 3.0, 6.0, 10.0, 14.0, 17.0])
    n = len(t)

    feature = LinearTrend()
    actual = feature(t, m, None)

    (slope, c), popt = curve_fit(f, t, m, absolute_sigma=False)
    chisq = np.sum((m - (t * slope + c)) ** 2) / (n - 2)

    desired = (slope, np.sqrt(popt[0][0]), np.sqrt(chisq))
    assert_allclose(actual, desired, rtol=1e-6)


def test_lintrend_2():  # check if sigma is used in least squares
    m = np.linspace(20, 25, 15)
    t = np.linspace(0, 10, 15)
    sigma = np.linspace(0, 1, 15)

    feature = LinearTrend()
    actual = feature(t, m, None)
    desired = feature(t, m, sigma)
    assert_allclose(actual, desired)
