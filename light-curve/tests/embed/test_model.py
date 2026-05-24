"""Tests for SingleBandModel and ExplicitMultiBandModel band routing logic."""

from dataclasses import dataclass, field

import numpy as np
import pytest

from light_curve.embed import ATCAT, Astromer1
from light_curve.embed.input_tensors import InputTensors, concat_input_tensors
from light_curve.embed.model import ImplicitMultiBandModel

# ---------------------------------------------------------------------------
# Minimal concrete ImplicitMultiBandModel for unit testing
# ---------------------------------------------------------------------------


@dataclass
class _TwoFieldTensors(InputTensors):
    vals: np.ndarray = field(kw_only=True)
    extra: np.ndarray = field(kw_only=True)


class _FakeImplicit(ImplicitMultiBandModel):
    """Two-band model: band 0 → seq_size=3, band 1 → seq_size=5."""

    n_model_bands: int = 2
    seq_sizes: list[int] = [3, 5]
    model_outputs: frozenset[str] = frozenset({"mean"})
    hf_repo = None

    def __init__(self, **kwargs):
        super().__init__(session=None, reduction="beginning", **kwargs)

    def __call__(self, *arrays, band):
        return super().__call__(*arrays, band=band)

    def preprocess_single_band(self, vals, *, band_idx, seq_size):
        vals = np.asarray(vals, dtype=np.float32)
        sequences, mask = self.reduction.preprocess_lc(vals, seq_size=seq_size)
        extra = np.full_like(sequences, float(band_idx))
        return _TwoFieldTensors(bool_mask=mask, vals=sequences[:, :, np.newaxis], extra=extra[:, :, np.newaxis])

    def predict_tensors(self, tensors):
        return np.zeros((tensors.vals.shape[0], 1, 4))


def _make_two_band_lc(n0=10, n1=12, seed=0):
    rng = np.random.default_rng(seed)
    vals = rng.uniform(0, 1, n0 + n1).astype(np.float32)
    band = np.array([0] * n0 + [1] * n1)
    return vals, band


# ---------------------------------------------------------------------------
# ImplicitMultiBandModel.preprocess_lc unit tests
# ---------------------------------------------------------------------------


def test_implicit_band_splitting():
    """preprocess_lc splits arrays by band index correctly."""
    model = _FakeImplicit()
    vals, band = _make_two_band_lc(n0=10, n1=12)
    tensors = model.preprocess_lc(vals, band)

    # seq_size 3 + 5 = 8
    assert tensors.vals.shape == (1, 8, 1)
    assert tensors.extra.shape == (1, 8, 1)
    assert tensors.bool_mask.shape == (1, 8)


def test_implicit_band_info_values():
    """extra field should be 0.0 for band-0 positions and 1.0 for band-1 positions."""
    model = _FakeImplicit()
    vals, band = _make_two_band_lc(n0=10, n1=12)
    tensors = model.preprocess_lc(vals, band)

    extra = tensors.extra[0, :, 0]
    # first 3 positions → band 0, next 5 → band 1
    np.testing.assert_array_equal(extra[:3], 0.0)
    np.testing.assert_array_equal(extra[3:], 1.0)


def test_implicit_mask_shape_and_values():
    """bool_mask has correct shape; True for real obs, False for padding."""
    model = _FakeImplicit()
    vals, band = _make_two_band_lc(n0=2, n1=3)  # shorter than seq_sizes → padding
    tensors = model.preprocess_lc(vals, band)

    mask = tensors.bool_mask[0]  # (8,)
    # band 0: seq_size=3, 2 real → mask[:2]=True, mask[2]=False
    np.testing.assert_array_equal(mask[:2], True)
    np.testing.assert_array_equal(mask[2:3], False)
    # band 1: seq_size=5, 3 real → mask[3:6]=True, mask[6:8]=False
    np.testing.assert_array_equal(mask[3:6], True)
    np.testing.assert_array_equal(mask[6:8], False)


def test_implicit_mismatched_subsamples_raises():
    """Different n_subsamples per band raises ValueError."""
    from light_curve.embed.reduction import NonOverlappingWindows

    model = _FakeImplicit.__new__(_FakeImplicit)
    ImplicitMultiBandModel.__init__(model, session=None, reduction=NonOverlappingWindows())

    # band 0 has 6 obs → ceil(6/3)=2 windows; band 1 has 5 obs → ceil(5/5)=1 window → mismatch
    vals = np.ones(11, dtype=np.float32)
    band = np.array([0] * 6 + [1] * 5)
    with pytest.raises(ValueError, match="subsamples"):
        model.preprocess_lc(vals, band)


def test_implicit_string_band_groups():
    """band_groups dict maps string labels to integer indices."""
    model = _FakeImplicit(band_groups={"g": 0, "r": 1})
    vals = np.ones(15, dtype=np.float32)
    band = np.array(["g"] * 8 + ["r"] * 7)
    tensors = model.preprocess_lc(vals, band)
    assert tensors.vals.shape == (1, 8, 1)


# ---------------------------------------------------------------------------
# concat_input_tensors axis parameter
# ---------------------------------------------------------------------------


def test_concat_axis0():
    """axis=0 stacks tensors along the first axis (subsample dimension)."""
    t1 = _TwoFieldTensors(bool_mask=np.ones((2, 4), dtype=bool), vals=np.zeros((2, 4, 1)), extra=np.zeros((2, 4, 1)))
    t2 = _TwoFieldTensors(bool_mask=np.ones((3, 4), dtype=bool), vals=np.zeros((3, 4, 1)), extra=np.zeros((3, 4, 1)))
    out = concat_input_tensors([t1, t2], axis=0)
    assert out.vals.shape == (5, 4, 1)


def test_concat_axis1():
    """axis=1 concatenates tensors along the sequence dimension."""
    t1 = _TwoFieldTensors(bool_mask=np.ones((2, 3), dtype=bool), vals=np.zeros((2, 3, 1)), extra=np.zeros((2, 3, 1)))
    t2 = _TwoFieldTensors(bool_mask=np.ones((2, 5), dtype=bool), vals=np.zeros((2, 5, 1)), extra=np.zeros((2, 5, 1)))
    out = concat_input_tensors([t1, t2], axis=1)
    assert out.vals.shape == (2, 8, 1)
    assert out.bool_mask.shape == (2, 8)


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
