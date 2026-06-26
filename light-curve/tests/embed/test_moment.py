"""Tests for the MOMENT-1 embedding models (small/base/large)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pyarrow.parquet as pq
import pytest

from light_curve.embed import Moment1

PREP_MODELS_DIR = Path(__file__).parent.parent / "prep-models" / "models"

# MOMENT always emits a fixed 512-step context split into 64 patches of 8.
N_PATCHES = 64


@dataclass(frozen=True)
class MomentConfig:
    name: str
    make_model: Callable
    test_parquet: Path
    embed_dim: int


CONFIGS = [
    MomentConfig(
        name=f"moment1-{size}",
        make_model=(lambda size: lambda **kw: Moment1.from_hf(size=size, **kw))(size),
        test_parquet=PREP_MODELS_DIR / "moment" / "out" / size / "test-data" / f"moment1-{size}_test.parquet",
        embed_dim=dim,
    )
    for size, dim in [("small", 512), ("base", 768), ("large", 1024)]
]


@pytest.fixture(scope="session", params=CONFIGS, ids=[c.name for c in CONFIGS])
def config(request) -> MomentConfig:
    return request.param


@pytest.fixture(scope="session")
def data_table(config):
    return pq.read_table(config.test_parquet).to_pydict()


def _mags(lc: list[dict]) -> np.ndarray:
    return np.array([obs["mag"] for obs in lc], dtype=np.float32)


# ---------------------------------------------------------------------------
# Basic tests (only the smallest model, to limit downloads)
# ---------------------------------------------------------------------------


def test_from_hf_invalid_output():
    """from_hf() raises ValueError for unknown output names."""
    with pytest.raises(ValueError, match="Unknown output"):
        Moment1.from_hf(size="small", output="nonexistent")


def test_from_hf_invalid_size():
    """Moment1.from_hf() raises ValueError for unknown sizes."""
    with pytest.raises(ValueError, match="Unknown size"):
        Moment1.from_hf(size="nonexistent")


def test_mean_shape():
    """The 'mean' output has shape (1, 1, 1, embed_dim)."""
    model = Moment1.from_hf(size="small", output="mean")
    rng = np.random.default_rng(0)
    mag = rng.normal(18.0, 0.2, 137).astype(np.float32)
    embedding = model(mag)
    assert embedding.shape == (1, 1, 1, 512)
    assert np.all(np.isfinite(embedding))


def test_sequence_shape():
    """The 'sequence' output has shape (1, 1, 64, embed_dim), regardless of input length."""
    model = Moment1.from_hf(size="small", output="sequence")
    rng = np.random.default_rng(1)
    mag = rng.normal(18.0, 0.2, 100).astype(np.float32)
    embedding = model(mag)
    assert embedding.shape == (1, 1, N_PATCHES, 512)
    assert np.all(np.isfinite(embedding))


def test_sequence_rejects_multiwindow():
    """The 'sequence' output rejects multi-window reductions."""
    model = Moment1.from_hf(size="small", output="sequence", reduction="non-overlapping-windows")
    rng = np.random.default_rng(2)
    mag = rng.normal(18.0, 0.2, 1100).astype(np.float32)  # > 512 → multiple windows
    with pytest.raises(ValueError, match="single-subsample"):
        model(mag)


# ---------------------------------------------------------------------------
# Reference tests (all three sizes)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row_idx", range(10))
def test_mean_matches_reference(config, data_table, row_idx):
    """'mean' output matches the reference embedding from the parquet."""
    model = config.make_model(output="mean")
    mag = _mags(data_table["lightcurve"][row_idx])
    expected = np.array(data_table["embedding_mean"][row_idx])

    embedding = model(mag)
    assert embedding.shape == (1, 1, 1, config.embed_dim)
    assert np.all(np.isfinite(embedding))

    emb_vec = embedding[0, 0, 0].astype(np.float64)
    ref_vec = expected.astype(np.float64)
    cos_sim = np.dot(emb_vec, ref_vec) / (np.linalg.norm(emb_vec) * np.linalg.norm(ref_vec))
    assert cos_sim > 0.99, f"cosine similarity {cos_sim:.6f} < 0.99"
