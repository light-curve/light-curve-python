"""Tests for the ATCAT multiband embedding model."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

PREP_MODELS_DIR = Path(__file__).parent.parent / "prep-models" / "models"
ATCAT_PARQUET = PREP_MODELS_DIR / "atcat" / "out" / "test-data" / "atcat_test.parquet"

# Maps LSST string band labels to the integer indices the ATCAT model expects
LSST_BAND_GROUPS = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "Y": 5}


def _flatten_lc(lc_by_band: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct a time-sorted flat sequence from per-band parquet storage."""
    times, fluxes, flux_errs, bands = [], [], [], []
    for entry in lc_by_band:
        n = len(entry["time"])
        times.extend(entry["time"])
        fluxes.extend(entry["flux"])
        flux_errs.extend(entry["flux_err"])
        bands.extend([entry["band"]] * n)
    times = np.asarray(times, dtype=np.float32)
    order = np.argsort(times, kind="stable")
    return (
        times[order],
        np.asarray(fluxes, dtype=np.float32)[order],
        np.asarray(flux_errs, dtype=np.float32)[order],
        np.asarray(bands)[order],
    )


@pytest.fixture(scope="session")
def atcat_data_table():
    return pq.read_table(ATCAT_PARQUET).to_pydict()


# ---------------------------------------------------------------------------
# Basic tests (no reference data needed)
# ---------------------------------------------------------------------------


def test_from_hf_invalid_output():
    """from_hf() raises ValueError for unknown output names."""
    import light_curve.embed as lce

    with pytest.raises(ValueError, match="Unknown output"):
        lce.ATCAT.from_hf(output="nonexistent")


def test_from_hf_shape():
    """from_hf() with band_groups returns a working model with correct output shape."""
    import light_curve.embed as lce

    model = lce.ATCAT.from_hf(output="last", reduction="beginning", band_groups=LSST_BAND_GROUPS)
    n = 50
    band_names = list(LSST_BAND_GROUPS.keys())
    time = np.linspace(0, 200, n, dtype=np.float32)
    flux = np.ones(n, dtype=np.float32)
    flux_err = np.full(n, 0.1, dtype=np.float32)
    band = np.array([band_names[i % len(band_names)] for i in range(n)])
    embedding = model(time, flux, flux_err, band)
    assert embedding.shape == (1, 1, 1, 384)
    assert np.all(np.isfinite(embedding))


# ---------------------------------------------------------------------------
# Reference tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row_idx", range(10))
def test_last_matches_reference(atcat_data_table, row_idx):
    """'last' output matches the reference embedding from the parquet."""
    import light_curve.embed as lce

    lc = atcat_data_table["lightcurve"][row_idx]
    time, flux, flux_err, band = _flatten_lc(lc)
    expected = np.array(atcat_data_table["embedding_last"][row_idx])

    model = lce.ATCAT.from_hf(output="last", reduction="beginning", band_groups=LSST_BAND_GROUPS, mag_zp=27.5)
    embedding = model(time, flux, flux_err, band)

    assert embedding.shape == (1, 1, 1, 384)
    assert np.all(np.isfinite(embedding))
    emb_vec = embedding[0, 0, 0].astype(np.float64)
    ref_vec = expected.astype(np.float64)
    cos_sim = np.dot(emb_vec, ref_vec) / (np.linalg.norm(emb_vec) * np.linalg.norm(ref_vec))
    assert cos_sim > 0.99, f"cosine similarity {cos_sim:.6f} < 0.99"
