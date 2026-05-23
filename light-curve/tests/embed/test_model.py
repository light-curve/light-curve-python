"""Tests for SingleBandModel and ExplicitMultiBandModel band routing logic."""

import numpy as np
import pytest

from light_curve.embed import ATCAT, Astromer1


def test_band_groups_invalid_mapped_value():
    """Mapping to an integer outside valid_model_bands raises ValueError."""
    with pytest.raises(ValueError, match="Invalid values found"):
        ATCAT.from_hf(output="last", band_groups={"g": 99})


def test_band_groups_duplicate_keys_in_list():
    """Duplicate input keys across list of dicts raise ValueError."""
    with pytest.raises(ValueError, match="Duplicate keys found"):
        ATCAT.from_hf(
            output="last",
            band_groups=[{"g": 1, "r": 2}, {"g": 3, "i": 4}],
        )


def test_band_groups_single_dict_stored(atcat_lsst_band_groups):
    """A valid single dict mapping is stored on the model."""
    model = ATCAT.from_hf(output="last", band_groups=atcat_lsst_band_groups)
    assert model.bands_groups == atcat_lsst_band_groups


def test_band_groups_list_of_dicts_creates_two_mappings():
    """A valid list of two disjoint dicts produces two band_mappings."""
    model = ATCAT.from_hf(output="last", band_groups=[{"g": 1, "r": 2}, {"i": 3, "z": 4}])
    assert len(model.band_mappings) == 2


def test_band_groups_none_sets_valid_model_bands():
    """With band_groups=None, defined_input_bands equals valid_model_bands."""
    model = ATCAT.from_hf(output="last")
    assert model.defined_input_bands == set(range(ATCAT.n_model_bands))


# ---------------------------------------------------------------------------
# Runtime tests: allow_extra_bands and output shape
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def atcat_last_model(atcat_lsst_band_groups):
    return ATCAT.from_hf(output="last", reduction="beginning", band_groups=atcat_lsst_band_groups)


def _make_lc(band_groups: dict, n: int = 60):
    rng = np.random.default_rng(0)
    band_names = list(band_groups.keys())
    time = np.sort(rng.uniform(0, 500, n)).astype(np.float32)
    flux = rng.normal(100, 10, n).astype(np.float32)
    flux_err = np.full(n, 5.0, dtype=np.float32)
    band = np.array([band_names[i % len(band_names)] for i in range(n)])
    return time, flux, flux_err, band


def test_allow_extra_bands_false_raises(atcat_lsst_band_groups):
    """Unknown band label raises ValueError when allow_extra_bands=False."""
    model = ATCAT.from_hf(
        output="last", reduction="beginning", band_groups=atcat_lsst_band_groups, allow_extra_bands=False
    )
    time, flux, flux_err, band = _make_lc(atcat_lsst_band_groups)
    band[0] = "X"  # unknown band
    with pytest.raises(ValueError, match="Invalid values found"):
        model(time, flux, flux_err, band)


def test_allow_extra_bands_true_ignores_unknown(atcat_lsst_band_groups):
    """Unknown band label is silently ignored when allow_extra_bands=True."""
    model = ATCAT.from_hf(
        output="last", reduction="beginning", band_groups=atcat_lsst_band_groups, allow_extra_bands=True
    )
    time, flux, flux_err, band = _make_lc(atcat_lsst_band_groups)
    band[0] = "X"  # unknown band — should not raise
    embedding = model(time, flux, flux_err, band)
    assert embedding.shape == (1, 1, 1, 384)
    assert np.all(np.isfinite(embedding))


def test_list_band_groups_output_shape(atcat_lsst_band_groups):
    """Two disjoint band_groups dicts produce n_band_groups=2 in the output."""
    band_groups = [{"g": 1, "r": 2, "i": 3}, {"z": 4, "Y": 5}]
    model = ATCAT.from_hf(output="last", reduction="beginning", band_groups=band_groups)
    time, flux, flux_err, band = _make_lc(atcat_lsst_band_groups)
    mask = np.isin(band, ["g", "r", "i", "z", "Y"])  # drop u
    embedding = model(time[mask], flux[mask], flux_err[mask], band[mask])
    assert embedding.shape == (2, 1, 1, 384)
    assert np.all(np.isfinite(embedding))


def test_integer_band_groups_none():
    """With band_groups=None, integer band indices 0-5 are passed directly."""
    model = ATCAT.from_hf(output="last", reduction="beginning")
    n = 60
    rng = np.random.default_rng(1)
    time = np.sort(rng.uniform(0, 500, n)).astype(np.float32)
    flux = rng.normal(100, 10, n).astype(np.float32)
    flux_err = np.full(n, 5.0, dtype=np.float32)
    band = np.array([i % 6 for i in range(n)])
    embedding = model(time, flux, flux_err, band)
    assert embedding.shape == (1, 1, 1, 384)
    assert np.all(np.isfinite(embedding))


# ---------------------------------------------------------------------------
# SingleBandModel: bands routing
# ---------------------------------------------------------------------------


def _make_single_band_lc(bands: list, n: int = 100):
    rng = np.random.default_rng(2)
    time = np.sort(rng.uniform(0, 500, n)).astype(np.float64)
    mag = rng.normal(15, 1, n).astype(np.float64)
    band = np.array([bands[i % len(bands)] for i in range(n)])
    return time, mag, band


def test_single_band_no_bands_shape():
    """bands=None treats the whole LC as one band → n_bands=1."""
    model = Astromer1.from_hf(output="mean", reduction="beginning")
    time, mag, _ = _make_single_band_lc(["g"])
    embedding = model(time, mag)
    assert embedding.shape == (1, 1, 1, 256)


def test_single_band_with_bands_shape():
    """bands=['g','r'] splits the LC → n_bands=2."""
    model = Astromer1.from_hf(output="mean", reduction="beginning", bands=["g", "r"])
    time, mag, band = _make_single_band_lc(["g", "r"])
    embedding = model(time, mag, band=band)
    assert embedding.shape == (2, 1, 1, 256)


def test_single_band_band_without_bands_raises():
    """Passing band= when bands=None raises ValueError."""
    model = Astromer1.from_hf(output="mean", reduction="beginning")
    time, mag, band = _make_single_band_lc(["g", "r"])
    with pytest.raises(ValueError):
        model(time, mag, band=band)


def test_single_band_missing_band_raises():
    """Omitting band= when bands is set raises ValueError."""

    model = Astromer1.from_hf(output="mean", reduction="beginning", bands=["g", "r"])
    time, mag, _ = _make_single_band_lc(["g", "r"])
    with pytest.raises(ValueError):
        model(time, mag)
