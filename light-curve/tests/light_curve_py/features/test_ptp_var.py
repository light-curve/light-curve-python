import numpy as np
import scipy.stats as st
from numpy.testing import assert_allclose

from light_curve.light_curve_py import PeakToPeakVar


def test_ptpvar():
    feature = PeakToPeakVar()
    rng = np.random.default_rng(0)
    n = 100
    lmd = 200
    t = np.arange(n)
    flux = rng.poisson(lmd, n)
    sigma = np.sqrt(flux)

    a1 = st.norm.ppf(q=(0.025) ** (1 / n), loc=n - np.sqrt(n), scale=np.sqrt(n + 0.5**2))
    b1 = st.norm.ppf(q=(0.975) ** (1 / n), loc=n - np.sqrt(n), scale=np.sqrt(n + 0.5**2))
    a2 = st.norm.ppf(q=((((0.025 - 1) * (-1)) ** (1 / n)) - 1) * (-1), loc=n + np.sqrt(n), scale=np.sqrt(n + 0.5**2))
    b2 = st.norm.ppf(q=((((0.975 - 1) * (-1)) ** (1 / n)) - 1) * (-1), loc=n + np.sqrt(n), scale=np.sqrt(n + 0.5**2))

    a = (a1 - b2) / (a1 + a2)
    b = (b1 - a2) / (b1 + b2)
    c = (a + b) / 2
    d = (b - a) / 2

    actual = feature(t, flux, sigma)
    desired = c
    atol = d
    assert_allclose(actual, desired, atol=atol)
