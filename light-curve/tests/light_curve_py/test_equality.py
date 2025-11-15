"""Tests for equality and hashing operators in Python features."""

import pytest

from light_curve.light_curve_py import (
    Amplitude,
    LinearFit,
    Mean,
    Median,
    StandardDeviation,
)


class TestEqualityOperators:
    """Test __eq__ operator for Python feature evaluators."""

    def test_same_feature_no_params_equal(self):
        """Two instances of the same feature with no parameters should be equal."""
        a1 = Amplitude()
        a2 = Amplitude()
        assert a1 == a2

    def test_same_feature_no_params_not_identical(self):
        """Two instances are equal but not identical."""
        a1 = Amplitude()
        a2 = Amplitude()
        assert a1 is not a2

    def test_different_features_not_equal(self):
        """Different feature types should not be equal."""
        a = Amplitude()
        m = Mean()
        assert a != m
        assert not (a == m)

    def test_feature_with_bands_equal(self):
        """Features with the same bands should be equal."""
        lf1 = LinearFit(bands=["g", "r"])
        lf2 = LinearFit(bands=["g", "r"])
        assert lf1 == lf2

    def test_feature_with_different_bands_not_equal(self):
        """Features with different bands should not be equal."""
        lf1 = LinearFit(bands=["g", "r"])
        lf2 = LinearFit(bands=["r", "g"])
        assert lf1 != lf2

    def test_feature_bands_vs_no_bands_not_equal(self):
        """Features with and without bands should not be equal."""
        lf1 = LinearFit()
        lf2 = LinearFit(bands=["g", "r"])
        assert lf1 != lf2


class TestHashOperator:
    """Test __hash__ operator for Python feature evaluators."""

    def test_same_feature_no_params_same_hash(self):
        """Two instances of the same feature should have the same hash."""
        a1 = Amplitude()
        a2 = Amplitude()
        assert hash(a1) == hash(a2)

    def test_different_features_different_hash(self):
        """Different features should have different hashes."""
        a = Amplitude()
        m = Mean()
        # Note: hash collisions are possible but unlikely for different features
        assert hash(a) != hash(m)

    def test_feature_with_bands_same_hash(self):
        """Features with the same bands should have the same hash."""
        lf1 = LinearFit(bands=["g", "r"])
        lf2 = LinearFit(bands=["g", "r"])
        assert hash(lf1) == hash(lf2)

    def test_feature_with_different_bands_different_hash(self):
        """Features with different bands should have different hashes."""
        lf1 = LinearFit(bands=["g", "r"])
        lf2 = LinearFit(bands=["r", "g"])
        assert hash(lf1) != hash(lf2)

    def test_hash_is_stable(self):
        """Hash should be stable across multiple calls."""
        a = Amplitude()
        h1 = hash(a)
        h2 = hash(a)
        h3 = hash(a)
        assert h1 == h2 == h3

    def test_features_can_be_used_in_set(self):
        """Features with proper hash can be used in sets."""
        a1 = Amplitude()
        a2 = Amplitude()
        m = Mean()

        # Create a set with features
        feature_set = {a1, m}
        assert len(feature_set) == 2

        # Add another Amplitude - should not increase size due to equality
        feature_set.add(a2)
        assert len(feature_set) == 2

        # Verify we can find features in the set
        assert a1 in feature_set
        assert a2 in feature_set
        assert m in feature_set

    def test_features_can_be_dict_keys(self):
        """Features with proper hash can be used as dictionary keys."""
        a1 = Amplitude()
        a2 = Amplitude()
        m = Mean()

        # Create a dict with features as keys
        feature_dict = {a1: "amplitude", m: "mean"}
        assert len(feature_dict) == 2

        # Access using equal feature instance
        assert feature_dict[a2] == "amplitude"

        # Update with equal feature - should update existing key
        feature_dict[a2] = "updated_amplitude"
        assert len(feature_dict) == 2
        assert feature_dict[a1] == "updated_amplitude"

    def test_features_with_bands_in_set(self):
        """Features with bands can be used in sets."""
        lf1 = LinearFit(bands=["g", "r"])
        lf2 = LinearFit(bands=["g", "r"])
        lf3 = LinearFit(bands=["r", "i"])

        feature_set = {lf1, lf2, lf3}
        # lf1 and lf2 are equal, so only 2 unique features
        assert len(feature_set) == 2
        assert lf1 in feature_set
        assert lf2 in feature_set
        assert lf3 in feature_set
