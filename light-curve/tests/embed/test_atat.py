"""Tests for the ATAT multiband embedding model."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from light_curve.embed import ATAT

PREP_MODELS_DIR = Path(__file__).parent.parent / "prep-models" / "models"
ATAT_PARQUET = PREP_MODELS_DIR / "atat" / "out" / "test-data" / "atat_test.parquet"


def _flatten_lc(lc_by_band: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct a time-sorted flat sequence from per-band parquet storage."""
    times, fluxes, bands = [], [], []
    for entry in lc_by_band:
        n = len(entry["time"])
        times.extend(entry["time"])
        fluxes.extend(entry["flux"])
        bands.extend([entry["band"]] * n)
    times = np.asarray(times, dtype=np.float32)
    order = np.argsort(times, kind="stable")
    return (
        times[order],
        np.asarray(fluxes, dtype=np.float32)[order],
        np.asarray(bands)[order],
    )


@pytest.fixture(scope="session")
def atat_data_table():
    return pq.read_table(ATAT_PARQUET).to_pydict()


# ---------------------------------------------------------------------------
# Basic tests (no reference data needed)
# ---------------------------------------------------------------------------


def test_from_hf_invalid_output():
    """from_hf() raises ValueError for unknown output names."""

    with pytest.raises(ValueError, match="Unknown output"):
        ATAT.from_hf(output="nonexistent")


def test_from_hf_shape(atat_lsst_band_groups):
    """from_hf() with band_groups returns a working model with correct output shape."""

    model = ATAT.from_hf(output="token", reduction="beginning", band_groups=atat_lsst_band_groups)
    n = 50
    band_names = list(atat_lsst_band_groups.keys())
    time = np.linspace(0, 200, n, dtype=np.float32)
    flux = np.ones(n, dtype=np.float32)
    band = np.array([band_names[i % len(band_names)] for i in range(n)])
    embedding = model(time, flux, band)
    assert embedding.shape == (1, 1, 1, 192)
    assert np.all(np.isfinite(embedding))


def test_from_hf_sequence_shape(atat_lsst_band_groups):
    """The 'sequence' output returns per-observation embeddings of length 65 * 6."""

    model = ATAT.from_hf(output="sequence", reduction="beginning", band_groups=atat_lsst_band_groups)
    n = 50
    band_names = list(atat_lsst_band_groups.keys())
    time = np.linspace(0, 200, n, dtype=np.float32)
    flux = np.ones(n, dtype=np.float32)
    band = np.array([band_names[i % len(band_names)] for i in range(n)])
    embedding = model(time, flux, band)
    assert embedding.shape == (1, 1, 65 * 6, 192)
    assert np.all(np.isfinite(embedding))


# ---------------------------------------------------------------------------
# Reference tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row_idx", range(10))
def test_token_matches_reference(atat_data_table, row_idx, atat_lsst_band_groups):
    """'token' output matches the reference embedding from the parquet."""

    lc = atat_data_table["lightcurve"][row_idx]
    time, flux, band = _flatten_lc(lc)
    expected = np.array(atat_data_table["embedding_token"][row_idx])

    model = ATAT.from_hf(output="token", reduction="beginning", band_groups=atat_lsst_band_groups, mag_zp=27.5)
    embedding = model(time, flux, band)

    assert embedding.shape == (1, 1, 1, 192)
    assert np.all(np.isfinite(embedding))
    emb_vec = embedding[0, 0, 0].astype(np.float64)
    ref_vec = expected.astype(np.float64)
    cos_sim = np.dot(emb_vec, ref_vec) / (np.linalg.norm(emb_vec) * np.linalg.norm(ref_vec))
    assert cos_sim > 0.99, f"cosine similarity {cos_sim:.6f} < 0.99"
