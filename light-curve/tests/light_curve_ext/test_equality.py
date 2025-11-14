"""Tests for feature equality and hashing operators."""

import pytest

import light_curve.light_curve_ext as lc


class TestEqualityOperators:
    """Test __eq__ and __ne__ operators for feature evaluators."""

    def test_same_feature_no_params_equal(self):
        """Two instances of the same feature with no parameters should be equal."""
        a1 = lc.Amplitude()
        a2 = lc.Amplitude()
        assert a1 == a2
        assert not (a1 != a2)

    def test_same_feature_no_params_not_identical(self):
        """Two instances are equal but not identical."""
        a1 = lc.Amplitude()
        a2 = lc.Amplitude()
        assert a1 is not a2

    def test_different_features_not_equal(self):
        """Different feature types should not be equal."""
        a = lc.Amplitude()
        m = lc.Mean()
        assert a != m
        assert not (a == m)

    def test_same_feature_with_same_params_equal(self):
        """Features with the same parameters should be equal."""
        b1 = lc.BeyondNStd(1.5)
        b2 = lc.BeyondNStd(1.5)
        assert b1 == b2

    def test_same_feature_with_different_params_not_equal(self):
        """Features with different parameters should not be equal."""
        b1 = lc.BeyondNStd(1.5)
        b2 = lc.BeyondNStd(2.0)
        assert b1 != b2
        assert not (b1 == b2)

    def test_extractor_with_same_features_equal(self):
        """Extractors with the same features in the same order should be equal."""
        e1 = lc.Extractor(lc.Amplitude(), lc.Mean())
        e2 = lc.Extractor(lc.Amplitude(), lc.Mean())
        assert e1 == e2

    def test_extractor_with_different_order_not_equal(self):
        """Extractors with features in different order should not be equal."""
        e1 = lc.Extractor(lc.Amplitude(), lc.Mean())
        e2 = lc.Extractor(lc.Mean(), lc.Amplitude())
        assert e1 != e2

    def test_extractor_with_different_features_not_equal(self):
        """Extractors with different features should not be equal."""
        e1 = lc.Extractor(lc.Amplitude(), lc.Mean())
        e2 = lc.Extractor(lc.Amplitude(), lc.StandardDeviation())
        assert e1 != e2


class TestHashOperator:
    """Test __hash__ operator for feature evaluators."""

    def test_same_feature_no_params_same_hash(self):
        """Two instances of the same feature should have the same hash."""
        a1 = lc.Amplitude()
        a2 = lc.Amplitude()
        assert hash(a1) == hash(a2)

    def test_different_features_different_hash(self):
        """Different features should have different hashes."""
        a = lc.Amplitude()
        m = lc.Mean()
        # Note: hash collisions are possible but unlikely for different features
        assert hash(a) != hash(m)

    def test_same_feature_with_same_params_same_hash(self):
        """Features with the same parameters should have the same hash."""
        b1 = lc.BeyondNStd(1.5)
        b2 = lc.BeyondNStd(1.5)
        assert hash(b1) == hash(b2)

    def test_same_feature_with_different_params_different_hash(self):
        """Features with different parameters should have different hashes."""
        b1 = lc.BeyondNStd(1.5)
        b2 = lc.BeyondNStd(2.0)
        assert hash(b1) != hash(b2)

    def test_hash_is_stable(self):
        """Hash should be stable across multiple calls."""
        a = lc.Amplitude()
        h1 = hash(a)
        h2 = hash(a)
        h3 = hash(a)
        assert h1 == h2 == h3

    def test_features_can_be_used_in_set(self):
        """Features with proper hash can be used in sets."""
        a1 = lc.Amplitude()
        a2 = lc.Amplitude()
        m = lc.Mean()

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
        a1 = lc.Amplitude()
        a2 = lc.Amplitude()
        m = lc.Mean()

        # Create a dict with features as keys
        feature_dict = {a1: "amplitude", m: "mean"}
        assert len(feature_dict) == 2

        # Access using equal feature instance
        assert feature_dict[a2] == "amplitude"

        # Update with equal feature - should update existing key
        feature_dict[a2] = "updated_amplitude"
        assert len(feature_dict) == 2
        assert feature_dict[a1] == "updated_amplitude"

    def test_extractor_hash_consistency(self):
        """Extractors with the same features should have the same hash."""
        e1 = lc.Extractor(lc.Amplitude(), lc.Mean())
        e2 = lc.Extractor(lc.Amplitude(), lc.Mean())
        assert hash(e1) == hash(e2)

        # Different order should have different hash
        e3 = lc.Extractor(lc.Mean(), lc.Amplitude())
        assert hash(e1) != hash(e3)
