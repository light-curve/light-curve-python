"""Tests for equality and hashing operators in Python features."""

import pytest

from light_curve.light_curve_py import (
    Amplitude,
    Bins,
    ColorOfMedian,
    Cusum,
    Eta,
    EtaE,
    InterPercentileRange,
    Kurtosis,
    LinearFit,
    LinearTrend,
    MagnitudePercentageRatio,
    MaximumSlope,
    Mean,
    MeanVariance,
    Median,
    MedianAbsoluteDeviation,
    MedianBufferRangePercentage,
    PercentAmplitude,
    PercentDifferenceMagnitudePercentile,
    ReducedChi2,
    Roms,
    Skew,
    StandardDeviation,
    StetsonK,
    WeightedMean,
)

try:
    from light_curve.light_curve_py.features.rainbow import RainbowFit
    RAINBOW_AVAILABLE = True
except ImportError:
    RAINBOW_AVAILABLE = False


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

    def test_multiple_simple_features_equal(self):
        """Test equality for multiple simple features."""
        features = [
            (Amplitude(), Amplitude()),
            (Mean(), Mean()),
            (Median(), Median()),
            (StandardDeviation(), StandardDeviation()),
            (Kurtosis(), Kurtosis()),
            (Skew(), Skew()),
            (MaximumSlope(), MaximumSlope()),
            (WeightedMean(), WeightedMean()),
            (MedianAbsoluteDeviation(), MedianAbsoluteDeviation()),
            (MeanVariance(), MeanVariance()),
            (ReducedChi2(), ReducedChi2()),
        ]
        for f1, f2 in features:
            assert f1 == f2, f"Failed for {f1.__class__.__name__}"

    def test_bins_with_parameters_equal(self):
        """Test Bins feature with parameters."""
        b1 = Bins(window=1.0, offset=0.0)
        b2 = Bins(window=1.0, offset=0.0)
        assert b1 == b2

        b3 = Bins(window=2.0, offset=0.0)
        assert b1 != b3

    def test_color_of_median_equal(self):
        """Test ColorOfMedian feature."""
        c1 = ColorOfMedian("g", "r")
        c2 = ColorOfMedian("g", "r")
        assert c1 == c2

        c3 = ColorOfMedian("r", "i")
        assert c1 != c3

    def test_features_with_numeric_params_equal(self):
        """Test features with numeric parameters."""
        # InterPercentileRange
        ipr1 = InterPercentileRange(quantile=0.25)
        ipr2 = InterPercentileRange(quantile=0.25)
        assert ipr1 == ipr2

        ipr3 = InterPercentileRange(quantile=0.1)
        assert ipr1 != ipr3

        # PercentAmplitude
        pa1 = PercentAmplitude()
        pa2 = PercentAmplitude()
        assert pa1 == pa2

        # MagnitudePercentageRatio
        mpr1 = MagnitudePercentageRatio(quantile_numerator=0.4, quantile_denominator=0.05)
        mpr2 = MagnitudePercentageRatio(quantile_numerator=0.4, quantile_denominator=0.05)
        assert mpr1 == mpr2

        mpr3 = MagnitudePercentageRatio(quantile_numerator=0.5, quantile_denominator=0.05)
        assert mpr1 != mpr3

    @pytest.mark.skipif(not RAINBOW_AVAILABLE, reason="Rainbow features not available")
    def test_rainbow_fit_equal(self):
        """Test RainbowFit equality."""
        band_wave_aa = {"g": 4770.0, "r": 6231.0}
        
        rf1 = RainbowFit.from_angstrom(band_wave_aa, with_baseline=False, temperature="sigmoid", bolometric="bazin")
        rf2 = RainbowFit.from_angstrom(band_wave_aa, with_baseline=False, temperature="sigmoid", bolometric="bazin")
        assert rf1 == rf2

    @pytest.mark.skipif(not RAINBOW_AVAILABLE, reason="Rainbow features not available")
    def test_rainbow_fit_different_params_not_equal(self):
        """Test RainbowFit with different parameters are not equal."""
        band_wave_aa = {"g": 4770.0, "r": 6231.0}
        
        rf1 = RainbowFit.from_angstrom(band_wave_aa, with_baseline=False, temperature="sigmoid", bolometric="bazin")
        rf2 = RainbowFit.from_angstrom(band_wave_aa, with_baseline=True, temperature="sigmoid", bolometric="bazin")
        assert rf1 != rf2

    @pytest.mark.skipif(not RAINBOW_AVAILABLE, reason="Rainbow features not available")
    def test_rainbow_fit_different_bands_not_equal(self):
        """Test RainbowFit with different bands are not equal."""
        band_wave_aa1 = {"g": 4770.0, "r": 6231.0}
        band_wave_aa2 = {"g": 4770.0, "r": 6231.0, "i": 7625.0}
        
        rf1 = RainbowFit.from_angstrom(band_wave_aa1, with_baseline=False)
        rf2 = RainbowFit.from_angstrom(band_wave_aa2, with_baseline=False)
        assert rf1 != rf2

    @pytest.mark.skipif(not RAINBOW_AVAILABLE, reason="Rainbow features not available")
    def test_rainbow_fit_different_temperature_not_equal(self):
        """Test RainbowFit with different temperature models are not equal."""
        band_wave_aa = {"g": 4770.0, "r": 6231.0}
        
        rf1 = RainbowFit.from_angstrom(band_wave_aa, temperature="sigmoid", bolometric="bazin")
        rf2 = RainbowFit.from_angstrom(band_wave_aa, temperature="constant", bolometric="bazin")
        assert rf1 != rf2

    @pytest.mark.skipif(not RAINBOW_AVAILABLE, reason="Rainbow features not available")
    def test_rainbow_fit_different_bolometric_not_equal(self):
        """Test RainbowFit with different bolometric models are not equal."""
        band_wave_aa = {"g": 4770.0, "r": 6231.0}
        
        rf1 = RainbowFit.from_angstrom(band_wave_aa, bolometric="bazin")
        rf2 = RainbowFit.from_angstrom(band_wave_aa, bolometric="sigmoid")
        assert rf1 != rf2


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

    def test_multiple_features_hash_consistency(self):
        """Test hash consistency for multiple features."""
        features = [
            Amplitude(),
            Mean(),
            Median(),
            StandardDeviation(),
            Kurtosis(),
            Skew(),
            MaximumSlope(),
            WeightedMean(),
        ]
        
        # All different features should have different hashes (with high probability)
        hashes = [hash(f) for f in features]
        assert len(set(hashes)) == len(features)

    # Note: Bins, ColorOfMedian, and RainbowFit are not hashable because they have
    # computed fields (extractor, median_feature) that are not part of the dataclass
    # fields and are themselves not hashable. These features can still use __eq__
    # but cannot be used in sets or as dict keys.

