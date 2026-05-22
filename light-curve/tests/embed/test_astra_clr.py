"""Tests for the AstraCLR multiband embedding model."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

PREP_MODELS_DIR = Path(__file__).parent.parent / "prep-models" / "models"
ASTRA_CLR_PARQUET = PREP_MODELS_DIR / "astra-clr" / "out" / "test-data" / "astra_clr_test.parquet"
ASTRA_CLR_ONNX = PREP_MODELS_DIR / "astra-clr" / "out" / "onnx" / "astra_clr.onnx"


# ---------------------------------------------------------------------------
# Preprocessing tests (no ONNX session needed)
# ---------------------------------------------------------------------------


def test_preprocess_shape():
    """preprocess_lc returns tensors with the correct fixed shapes."""
    import light_curve.embed as lce

    model = lce.AstraCLR(session=None)
    rng = np.random.default_rng(0)
    n = 300
    time = np.sort(rng.uniform(58000, 60500, n))
    mag = rng.uniform(17, 21, n).astype(np.float32)
    magerr = rng.uniform(0.01, 0.15, n).astype(np.float32)
    band = rng.choice(["g", "r", "i"], n)
    tensors = model.preprocess_lc(time, mag, magerr, band)
    assert tensors.input.shape == (1, 700, 1)
    assert tensors.times.shape == (1, 700, 1)
    assert tensors.band_info.shape == (1, 700, 1)
    assert tensors.mask.shape == (1, 700)
    assert tensors.bool_mask.shape == (1, 700)


def test_preprocess_mask_convention():
    """mask is 0 for real observations, 1 for padded (AstraCLR convention)."""
    import light_curve.embed as lce

    model = lce.AstraCLR(session=None)
    time = np.linspace(58000, 60000, 50)
    mag = np.ones(50, dtype=np.float32) * 18.0
    magerr = np.ones(50, dtype=np.float32) * 0.05
    band = np.array(["g"] * 50)
    tensors = model.preprocess_lc(time, mag, magerr, band)
    # First 50 positions in g-window (0-299) should be real → mask=0
    assert (tensors.mask[0, :50] == 0.0).all()
    # Rest should be padded → mask=1
    assert (tensors.mask[0, 50:] == 1.0).all()


def test_preprocess_weighted_mean_subtracted():
    """Magnitudes are inverse-variance-weighted mean-subtracted per band."""
    import light_curve.embed as lce

    model = lce.AstraCLR(session=None)
    rng = np.random.default_rng(42)
    n = 100
    time = np.sort(rng.uniform(58000, 60000, n))
    mag = rng.uniform(17, 21, n).astype(np.float32)
    magerr = rng.uniform(0.01, 0.15, n).astype(np.float32)
    band = np.array(["r"] * n)
    tensors = model.preprocess_lc(time, mag, magerr, band)
    # r-band occupies positions 300-649; first 100 are real
    r_mag = tensors.input[0, 300:400, 0]
    weights = magerr[:n] ** -2
    weighted_mean = float(np.average(mag[:n], weights=weights))
    np.testing.assert_allclose(r_mag, mag[:n] - weighted_mean, atol=1e-5)


# ---------------------------------------------------------------------------
# End-to-end tests against reference parquet
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def astra_clr_data():
    if not ASTRA_CLR_PARQUET.exists():
        pytest.skip(f"Reference parquet not found: {ASTRA_CLR_PARQUET}")
    return pq.read_table(ASTRA_CLR_PARQUET).to_pydict()


@pytest.fixture(scope="session")
def astra_clr_model():
    import onnxruntime as ort

    import light_curve.embed as lce

    if ASTRA_CLR_ONNX.exists():
        session = ort.InferenceSession(str(ASTRA_CLR_ONNX))
        return lce.AstraCLR(session=session)
    return lce.AstraCLR.from_hf()


def test_embedding_shape(astra_clr_data, astra_clr_model):
    """from_hf() returns embeddings with shape (1, 1, 1, 512)."""
    lc = astra_clr_data["lightcurve"][0]
    time = np.array([obs["mjd"] for obs in lc])
    mag = np.array([obs["mag"] for obs in lc], dtype=np.float32)
    magerr = np.array([obs["magerr"] for obs in lc], dtype=np.float32)
    band = np.array([obs["band"] for obs in lc])
    embedding = astra_clr_model(time, mag, magerr, band)
    assert embedding.shape == (1, 1, 1, 512)


def test_embedding_matches_reference(astra_clr_data, astra_clr_model):
    """Embeddings match the reference values from the prep-models test data."""
    cos_sims = []
    for lc, ref_emb in zip(astra_clr_data["lightcurve"], astra_clr_data["embedding_mean"]):
        time = np.array([obs["mjd"] for obs in lc])
        mag = np.array([obs["mag"] for obs in lc], dtype=np.float32)
        magerr = np.array([obs["magerr"] for obs in lc], dtype=np.float32)
        band = np.array([obs["band"] for obs in lc])
        emb = astra_clr_model(time, mag, magerr, band).squeeze()
        ref = np.array(ref_emb, dtype=np.float32)
        cos_sim = float(np.dot(emb, ref) / (np.linalg.norm(emb) * np.linalg.norm(ref)))
        cos_sims.append(cos_sim)
    assert np.mean(cos_sims) > 0.999, f"Mean cosine similarity too low: {np.mean(cos_sims):.4f}"
