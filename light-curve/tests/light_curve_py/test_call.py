import numpy as np
import pytest
from light_curve.light_curve_py import MaximumSlope


def test_normalize():
    t = [1, 4, 3, 8]
    m = [2, 15, 4, 3]
    feature = MaximumSlope()
    actual = feature(t, m, None, sorted=False)
    desired = 11.0
    assert actual == desired


def test_fill_zero_division():
    t = [1, 1, 3, 4]
    feature = MaximumSlope()
    actual = feature(t, t, None, sorted=False, fill_value=1.0)
    desired = 1.0
    assert actual == desired


def test_fill_inf_values():
    t = [0, 1e-300]
    m = [0, 1e300]
    feature = MaximumSlope()
    actual = feature(t, m, None, sorted=True, fill_value=1.0)
    desired = 1.0
    assert actual == desired


def test_non_unique_values():
    t = [1, 1, 3, 4]
    feature = MaximumSlope()
    with pytest.raises(ValueError):
        feature(t, t, None, sorted=False)


def test_non_sorted_values():
    t = [2, 1, 3, 4]
    feature = MaximumSlope()
    with pytest.raises(ValueError):
        feature(t, t, None)
