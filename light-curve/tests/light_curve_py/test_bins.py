import numpy as np

from light_curve.light_curve_py import Amplitude, Bins, Mean, Median


def test_bins_1():
    feature = Bins(window=1.0, offset=100)
    t = np.array([0.0, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 5.0])
    m = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    sigma = np.array([10.0, 5.0, 10.0, 5.0, 10.0, 5.0, 10.0, 5.0, 10.0, 5.0, 10.0])
    sigma = 1 / np.sqrt(sigma)

    new_t = np.array([0.5, 1.5, 2.5, 5.5])
    new_m = np.array([0.0, 2.0, 6.333333333333333, 10.0])
    new_sigma = np.sqrt(1 / np.array([10.0, 6.666666666666667, 7.5, 10.0]))
    actual = np.array(list(feature.transform(t, m, sigma)))
    desired = np.array([new_t, new_m, new_sigma])
    np.testing.assert_allclose(actual, desired)


def test_bins_2():
    feature = Bins(window=2.0, offset=1 / 3)
    t = np.array([0.5, 1.0, 1.1, 1.2, 5.0, 5.2, 10.0])
    m = np.array([0.8, 0.9, 1.0, 5.0, 3.0, 2.0, 1.0])
    sigma = np.array([0.2, 0.1, 0.05, 0.12, 0.5, 0.8, 0.1])

    new_t = np.array([1.333333, 5.333333, 9.333333])
    new_m = np.array([1.4420560747663553, 2.7191011235955056, 1.0])
    new_sigma = np.array([0.08203031124295959, 0.5996253511966891, 0.1])
    actual = np.array(list(feature.transform(t, m, sigma)))
    desired = np.array([new_t, new_m, new_sigma])
    np.testing.assert_allclose(actual, desired, rtol=1e-6)


def test_bins_3():
    feature = Bins(window=2.0, offset=-999.8)
    t = np.array([0.5, 1.0, 1.1, 1.2, 5.0, 5.2, 7.5, 10.0])
    m = np.array([0.8, 0.9, 1.0, 5.0, 3.0, 2.0, 1.0, 10.0])
    sigma = np.array([0.2, 0.1, 0.05, 0.12, 0.5, 0.8, 0.1, 0.02])

    new_t = np.array([1.2, 5.2, 7.2, 9.2])
    new_m = np.array([1.442056, 2.719101, 1.0, 10.0])
    new_sigma = np.array([0.08203, 0.599625, 0.1, 0.02])
    actual = np.array(list(feature.transform(t, m, sigma)))
    desired = np.array([new_t, new_m, new_sigma])
    np.testing.assert_allclose(actual, desired, rtol=1e-5)


def test_bins_size():
    feature = Bins((Mean(), Amplitude(), Median()), window=2.0, offset=1.0)
    assert feature.size == 3
