import numpy as np
import pytest

from light_curve.light_curve_py import FluxNNotDetBeforeFd, MagnitudeNNotDetBeforeFd


def test_flux():
    feature = FluxNNotDetBeforeFd(10)
    m = np.array([1, 2, 10, 40, 50])
    t = [1, 2, 3, 4, 5]
    sigma = np.array([0.4, 0.2, 0.01, 0.03, 0.02])
    actual = feature(t, m, sigma)
    desired = 2
    assert actual == desired


def test_flux_strictly_fainter():
    feature = FluxNNotDetBeforeFd(10, strictly_fainter=True)
    m = np.array([1, 2, 2, 40, 50])
    t = [1, 2, 3, 4, 5]
    sigma = np.array([0.12, 0.2, 0.01, 0.03, 0.02])
    actual = feature(t, m, sigma)
    desired = 1
    assert actual == desired


def test_magnitude():
    feature = MagnitudeNNotDetBeforeFd(-1)
    m = np.array([10, 1, 2, 40, 50])
    sigma = [-1, -1, -1, 0.2, 0.03]
    t = [1, 2, 3, 4, 5]
    actual = feature(t, m, sigma)
    desired = 3
    assert actual == desired


def test_magnitude_strictly_fainter():
    feature = MagnitudeNNotDetBeforeFd(-1, strictly_fainter=True)
    m = np.array([10, 1, 2, 9, 50])
    sigma = [-1, -1, -1, 0.2, 0.03]
    t = [1, 2, 3, 4, 5]
    actual = feature(t, m, sigma)
    desired = 1
    assert actual == desired


def test_without_non_detections():
    feature1 = MagnitudeNNotDetBeforeFd(-1, strictly_fainter=True)
    feature2 = FluxNNotDetBeforeFd(-1, strictly_fainter=True)

    m = np.array([10, 1, 2, 9, 50])
    sigma = [0.1, 0.1, 0.1, 0.2, 0.03]
    t = [1, 2, 3, 4, 5]

    with pytest.raises(ValueError):
        feature1(t, m, sigma)
    with pytest.raises(ValueError):
        feature2(t, m, sigma)
