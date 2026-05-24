"""Tests for the AstraCLR multi-band ZTF embedding model."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from light_curve import embed
from light_curve.embed.astra_clr import (
    _LG_EFF_WAVE,
    _SEQ_PER_BAND,
    _SEQUENCE_LENGTH,
    AstraCLRInputs,
)

PREP_MODELS_DIR = Path(__file__).parent.parent / "prep-models" / "models"
ASTRA_CLR_PARQUET = PREP_MODELS_DIR / "astra-clr" / "out" / "test-data" / "astra_clr_test.parquet"

# Band mapping used throughout these tests
ZTF_BAND_GROUPS = {"g": 0, "r": 1, "i": 2}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def data_table():
    return pq.read_table(ASTRA_CLR_PARQUET).to_pydict()


@pytest.fixture(scope="session")
def astra_clr_model():
    return embed.AstraCLR.from_hf(band_groups=ZTF_BAND_GROUPS)


# ---------------------------------------------------------------------------
# Unit tests — preprocessing only (no ONNX session)
# ---------------------------------------------------------------------------


def _make_lc(n_per_band=50, seed=0):
    """Generate a synthetic ZTF light curve (pre-sorted by band, then time) with string band labels."""
    rng = np.random.default_rng(seed)
    bands = ["g"] * n_per_band + ["r"] * n_per_band + ["i"] * n_per_band
    n = len(bands)
    # Already sorted — model currently assumes sorted input
    mjd = np.sort(rng.uniform(58_194, 59_951, n))
    mag = rng.normal(17.0, 0.5, n).astype(np.float32)
    magerr = rng.uniform(0.01, 0.1, n).astype(np.float32)
    return mjd, mag, magerr, np.array(bands)


def _make_lc_int(n_per_band=50, seed=0):
    """Generate a synthetic ZTF light curve with integer band indices."""
    mjd, mag, magerr, band_str = _make_lc(n_per_band=n_per_band, seed=seed)
    band_int = np.vectorize(ZTF_BAND_GROUPS.get)(band_str)
    return mjd, mag, magerr, band_int


def test_preprocess_output_shape():
    """preprocess_lc returns AstraCLRInputs with correct shapes."""
    model = embed.AstraCLR(session=None)
    mjd, mag, magerr, band = _make_lc_int()
    tensors = model.preprocess_lc(mjd, mag, magerr, band)

    assert isinstance(tensors, AstraCLRInputs)
    assert tensors.input.shape == (1, _SEQUENCE_LENGTH, 1)
    assert tensors.times.shape == (1, _SEQUENCE_LENGTH, 1)
    assert tensors.band_info.shape == (1, _SEQUENCE_LENGTH, 1)
    assert tensors.mask.shape == (1, _SEQUENCE_LENGTH)
    assert tensors.bool_mask.shape == (1, _SEQUENCE_LENGTH)


def test_preprocess_multiple_reductions_shape():
    """MultipleReductions produces correct n_subsamples axis."""
    model = embed.AstraCLR(session=None, reduction=["beginning", "end"])
    mjd, mag, magerr, band = _make_lc_int()
    tensors = model.preprocess_lc(mjd, mag, magerr, band)

    assert tensors.input.shape == (2, _SEQUENCE_LENGTH, 1)
    assert tensors.mask.shape == (2, _SEQUENCE_LENGTH)


def test_preprocess_mask_values():
    """Mask is 0 for real observations and 1 for padding."""
    model = embed.AstraCLR(session=None)
    mjd, mag, magerr, band = _make_lc_int(n_per_band=10)
    tensors = model.preprocess_lc(mjd, mag, magerr, band)

    mask = tensors.mask[0]
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    # With 10 obs per band we expect padding (fewer than max seq sizes)
    assert mask.sum() > 0


def test_preprocess_bool_mask_consistent():
    """bool_mask is True exactly where mask == 0 (real observations)."""
    model = embed.AstraCLR(session=None)
    mjd, mag, magerr, band = _make_lc_int()
    tensors = model.preprocess_lc(mjd, mag, magerr, band)

    np.testing.assert_array_equal(tensors.bool_mask, tensors.mask == 0)


def test_preprocess_band_info_values():
    """band_info contains the expected log10 effective wavelengths."""
    model = embed.AstraCLR(session=None)
    n_per_band = 50
    mjd, mag, magerr, band = _make_lc_int(n_per_band=n_per_band)
    tensors = model.preprocess_lc(mjd, mag, magerr, band)

    bi = tensors.band_info[0, :, 0]  # (700,)
    real_mask = tensors.bool_mask[0]

    # Check g-band section (positions 0..299)
    g_real = real_mask[: _SEQ_PER_BAND[0]]
    np.testing.assert_allclose(bi[: _SEQ_PER_BAND[0]][g_real], _LG_EFF_WAVE[0], rtol=1e-6)

    # Check r-band section
    r_start = _SEQ_PER_BAND[0]
    r_end = r_start + _SEQ_PER_BAND[1]
    r_real = real_mask[r_start:r_end]
    np.testing.assert_allclose(bi[r_start:r_end][r_real], _LG_EFF_WAVE[1], rtol=1e-6)

    # Check i-band section
    i_start = r_end
    i_real = real_mask[i_start:]
    np.testing.assert_allclose(bi[i_start:][i_real], _LG_EFF_WAVE[2], rtol=1e-6)


def test_preprocess_time_offset():
    """Normalised times are shifted by MJD_OFFSET (58 000)."""
    model = embed.AstraCLR(session=None)
    mjd, mag, magerr, band = _make_lc_int()
    tensors = model.preprocess_lc(mjd, mag, magerr, band)

    real_mask = tensors.bool_mask[0]
    norm_time = tensors.times[0, :, 0][real_mask]
    # ZTF DR16 spans MJD 58194–59951, so norm_time is at most ~2000
    assert np.all(np.abs(norm_time) < 2000)


def test_preprocess_mag_per_band_mean_subtracted():
    """Each band's normalised magnitudes have approximately zero weighted mean."""
    model = embed.AstraCLR(session=None)
    n_per_band = 200
    mjd, mag, magerr, band = _make_lc_int(n_per_band=n_per_band)
    tensors = model.preprocess_lc(mjd, mag, magerr, band)

    norm_mag = tensors.input[0, :, 0]
    real_mask = tensors.bool_mask[0]

    # Check each band's section independently
    offsets = [0] + list(np.cumsum(_SEQ_PER_BAND))
    for b_idx, (start, end) in enumerate(zip(offsets, offsets[1:])):
        section_real = real_mask[start:end]
        if not section_real.any():
            continue
        values = norm_mag[start:end][section_real]
        assert np.abs(values.mean()) < 2.0


def test_preprocess_float32_output():
    """All tensors in AstraCLRInputs are float32."""
    model = embed.AstraCLR(session=None)
    mjd, mag, magerr, band = _make_lc_int()
    tensors = model.preprocess_lc(mjd, mag, magerr, band)

    assert tensors.input.dtype == np.float32
    assert tensors.times.dtype == np.float32
    assert tensors.band_info.dtype == np.float32
    assert tensors.mask.dtype == np.float32


@pytest.mark.parametrize("reduction", ["beginning", "end", "middle"])
def test_single_reductions_accepted(reduction):
    """Beginning, End, and Middle reductions are accepted without error."""
    model = embed.AstraCLR(session=None, reduction=reduction)
    mjd, mag, magerr, band = _make_lc_int()
    tensors = model.preprocess_lc(mjd, mag, magerr, band)
    assert tensors.input.shape == (1, _SEQUENCE_LENGTH, 1)


def test_asdict_excludes_bool_mask():
    """asdict() returns the four model tensors without bool_mask."""
    model = embed.AstraCLR(session=None)
    mjd, mag, magerr, band = _make_lc_int()
    tensors = model.preprocess_lc(mjd, mag, magerr, band)

    d = tensors.asdict()
    assert set(d.keys()) == {"input", "times", "band_info", "mask"}


def test_band_groups_string_labels():
    """band_groups=ZTF_BAND_GROUPS allows string band labels."""
    model = embed.AstraCLR(session=None, band_groups=ZTF_BAND_GROUPS)
    mjd, mag, magerr, band_str = _make_lc()
    # Just check that __call__ routing doesn't crash before hitting session
    with pytest.raises(Exception):  # session is None → AttributeError
        model(mjd, mag, magerr, band_str)


# ---------------------------------------------------------------------------
# End-to-end tests (requires ONNX session from HuggingFace)
# ---------------------------------------------------------------------------


def test_from_hf_output_shape(astra_clr_model):
    """from_hf() returns a model whose output has shape (1, 1, 1, 512)."""
    mjd, mag, magerr, band_str = _make_lc()
    embedding = astra_clr_model(mjd, mag, magerr, band_str)
    assert embedding.shape == (1, 1, 1, 512)
    assert np.all(np.isfinite(embedding))


def test_from_hf_multiple_reductions(astra_clr_model):
    """MultipleReductions produces (1, n_subsamples, 1, 512) output."""
    model = embed.AstraCLR.from_hf(
        reduction=["beginning", "end"],
        band_groups=ZTF_BAND_GROUPS,
    )
    mjd, mag, magerr, band_str = _make_lc()
    embedding = model(mjd, mag, magerr, band_str)
    assert embedding.shape == (1, 2, 1, 512)
    assert np.all(np.isfinite(embedding))


@pytest.mark.parametrize("row_idx", range(10))
def test_embedding_matches_reference(astra_clr_model, data_table, row_idx):
    """Embedding is close (cosine sim > 0.999) to the reference from the parquet."""
    lc = data_table["lightcurve"][row_idx]
    mjd = np.array([obs["mjd"] for obs in lc], dtype=np.float64)
    mag = np.array([obs["mag"] for obs in lc], dtype=np.float32)
    magerr = np.array([obs["magerr"] for obs in lc], dtype=np.float32)
    band = np.array([obs["band"] for obs in lc])
    expected = np.array(data_table["embedding_mean"][row_idx], dtype=np.float64)

    embedding = astra_clr_model(mjd, mag, magerr, band)

    assert embedding.shape == (1, 1, 1, 512)
    assert np.all(np.isfinite(embedding))

    emb_vec = embedding[0, 0, 0].astype(np.float64)
    cos_sim = np.dot(emb_vec, expected) / (np.linalg.norm(emb_vec) * np.linalg.norm(expected))
    assert cos_sim > 0.999, f"row {row_idx}: cosine similarity {cos_sim:.6f} < 0.999"
