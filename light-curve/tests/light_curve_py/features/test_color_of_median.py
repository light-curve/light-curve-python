import pytest

from light_curve.light_curve_py import ColorOfMedian


def test_color_of_median():
    m = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    band = ["g", "g", "g", "r", "r", "r", "i"]

    feature = ColorOfMedian("g", "r")
    actual = feature(t=m, m=m, sigma=None, band=band)
    desired = -3.0
    assert actual == desired


def test_color_of_median_wrong_number_of_args():
    with pytest.raises(TypeError):
        _feature = ColorOfMedian()

    with pytest.raises(TypeError):
        _feature = ColorOfMedian("gr")

    with pytest.raises(TypeError):
        _feature = ColorOfMedian("g", "r", "i")
