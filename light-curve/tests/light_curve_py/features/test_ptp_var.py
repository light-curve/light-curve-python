import numpy as np
from numpy.testing import assert_allclose

from light_curve.light_curve_py import PeakToPeakVar


def test_ptpvar():
    feature = PeakToPeakVar()
    rng = np.random.default_rng(0)
    n = 100
    m = 10000
    lmd = 200
    t = np.arange(n)
    flux_list = np.random.poisson(lmd, (m, n))
    ptp_list = [PeakToPeakVar(t, flux_list[i], np.sqrt(flux_list[i])) for i in range(m)]
    flux = rng.poisson(lmd, n)
    sigma = np.sqrt(flux)
    actual = feature(t, flux, sigma)
    desired = (np.quantile(ptp_list, 0.025) + np.quantile(ptp_list, 0.975)) / 2
    atol = desired - np.quantile(ptp_list, 0.025)
    assert_allclose(actual, desired, atol=atol)