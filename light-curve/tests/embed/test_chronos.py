"""Tests for the Chronos-family embedding models (Chronos 2 and Chronos-Bolt)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pyarrow.parquet as pq
import pytest

from light_curve.embed import Chronos2, ChronosBolt

PREP_MODELS_DIR = Path(__file__).parent.parent / "prep-models" / "models"


@dataclass(frozen=True)
class ChronosConfig:
    name: str
    make_model: Callable
    test_parquet: Path
    embed_dim: int


CONFIGS = [
    ChronosConfig(
        name="chronos2",
        make_model=lambda **kw: Chronos2.from_hf(**kw),
        test_parquet=PREP_MODELS_DIR / "chronos2" / "out" / "test-data" / "chronos2_test.parquet",
        embed_dim=768,
    ),
    *[
        ChronosConfig(
            name=f"chronos-bolt-{size}",
            make_model=(lambda size: lambda **kw: ChronosBolt.from_hf(size=size, **kw))(size),
            test_parquet=PREP_MODELS_DIR
            / "chronos-bolt"
            / "out"
            / size
            / "test-data"
            / f"chronos-bolt-{size}_test.parquet",
            embed_dim=dim,
        )
        for size, dim in [("tiny", 256), ("mini", 384), ("small", 512), ("base", 768)]
    ],
]


@pytest.fixture(scope="session", params=CONFIGS, ids=[c.name for c in CONFIGS])
def config(request) -> ChronosConfig:
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
        ChronosBolt.from_hf(size="tiny", output="nonexistent")


def test_from_hf_invalid_size():
    """ChronosBolt.from_hf() raises ValueError for unknown sizes."""
    with pytest.raises(ValueError, match="Unknown size"):
        ChronosBolt.from_hf(size="nonexistent")


def test_mean_shape():
    """The 'mean' output has shape (1, 1, 1, embed_dim)."""
    model = ChronosBolt.from_hf(size="tiny", output="mean")
    rng = np.random.default_rng(0)
    mag = rng.normal(18.0, 0.2, 137).astype(np.float32)
    embedding = model(mag)
    assert embedding.shape == (1, 1, 1, 256)
    assert np.all(np.isfinite(embedding))


def test_sequence_shape():
    """The 'sequence' output has shape (1, 1, n_patches, embed_dim) with n_patches = ceil(n/16)."""
    model = ChronosBolt.from_hf(size="tiny", output="sequence")
    rng = np.random.default_rng(1)
    n = 100
    mag = rng.normal(18.0, 0.2, n).astype(np.float32)
    embedding = model(mag)
    assert embedding.shape == (1, 1, (n + 15) // 16, 256)
    assert np.all(np.isfinite(embedding))


# ---------------------------------------------------------------------------
# Reference tests (all five models)
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
