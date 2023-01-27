import numpy as np
from numpy.testing import assert_allclose
from scipy import signal

from light_curve.light_curve_py import StetsonK


def test_stetsonk_1():
    feature = StetsonK()
    x = np.linspace(0.0, 2 * np.pi, 2000)
    sin = np.sin(x)
    error = np.ones(2000)
    actual = feature(x, sin, error)
    desired = np.sqrt(8) / np.pi
    assert_allclose(actual, desired, rtol=1e-03)


def test_stetsonk_2():
    feature = StetsonK()
    x = np.linspace(0.0, 1.0, 1000)
    sawtooth = signal.sawtooth(2 * np.pi * 5 * x)
    error = np.ones(1000)
    actual = feature(x, sawtooth, error)
    desired = np.sqrt(12) / 4
    assert_allclose(actual, desired, rtol=1e-03)
