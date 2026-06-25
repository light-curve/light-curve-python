"""Tests for Astromer1 and Astromer2 embedding models."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from light_curve import embed

PREP_MODELS_DIR = Path(__file__).parent.parent / "prep-models" / "models"


@dataclass(frozen=True)
class AstromerConfig:
    model_name: str
    test_parquet: Path
    ref_reduction: str
    cos_sim_threshold: float


CONFIGS = [
    AstromerConfig(
        model_name="Astromer1",
        test_parquet=PREP_MODELS_DIR / "astromer1" / "out" / "test-data" / "astromer1_test.parquet",
        ref_reduction="beginning",
        cos_sim_threshold=0.97,
    ),
    AstromerConfig(
        model_name="Astromer1ZTF",
        test_parquet=PREP_MODELS_DIR / "astromer1-ztfdr20" / "out" / "test-data" / "astromer1_ztfdr20_test.parquet",
        ref_reduction="beginning",
        cos_sim_threshold=0.97,
    ),
    AstromerConfig(
        model_name="Astromer2",
        test_parquet=PREP_MODELS_DIR / "astromer2" / "out" / "test-data" / "astromer2_test.parquet",
        ref_reduction="beginning",
        cos_sim_threshold=0.999,
    ),
]


@pytest.fixture(
    scope="session",
    params=CONFIGS,
    ids=[c.model_name.lower() for c in CONFIGS],
)
def config(request) -> AstromerConfig:
    return request.param


@pytest.fixture(scope="session")
def data_table(config):
    return pq.read_table(config.test_parquet).to_pydict()


# ---------------------------------------------------------------------------
# Preprocessing tests (no ONNX session needed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_class_name", ["Astromer1", "Astromer1ZTF", "Astromer2"])
def test_astromer_preprocess_normalises(model_class_name):
    """preprocess_lc produces zero-mean time and mag for each window."""

    model_class = getattr(embed, model_class_name)
    model = model_class(session=None, reduction="non-overlapping-windows")
    time = np.linspace(0, 100, 200)
    mag = np.linspace(10, 15, 200)
    tensors = model.preprocess_lc(time, mag)
    valid = tensors.bool_mask[0]
    assert abs(tensors.times[0, valid, 0].mean()) < 1e-5
    assert abs(tensors.input[0, valid, 0].mean()) < 1e-5


# ---------------------------------------------------------------------------
# End-to-end tests
# ---------------------------------------------------------------------------


def test_sequence_output_long_lc(config):
    """NonOverlappingWindows + 'sequence' output yields (1, 1, 200, 256) shape."""

    model_class = getattr(embed, config.model_name)
    model = model_class.from_hf(output="sequence", reduction="non-overlapping-windows")

    time = np.linspace(0, 1000, 350)
    mag = np.ones(350)
    embedding = model(time, mag)

    assert embedding.shape == (1, 1, 200, 256)
    assert np.all(np.isfinite(embedding))


def test_from_hf_shape(config):
    """from_hf() returns a working model with correct output shape."""

    model_class = getattr(embed, config.model_name)
    model = model_class.from_hf(output="mean")
    time = np.linspace(0, 100, 50)
    mag = np.ones(50)
    embedding = model(time, mag)
    assert embedding.shape == (1, 1, 1, 256)
    assert np.all(np.isfinite(embedding))


def test_from_hf_invalid_output(config):
    """from_hf() raises ValueError for unknown output names."""

    model_class = getattr(embed, config.model_name)
    with pytest.raises(ValueError, match="Unknown output"):
        model_class.from_hf(output="nonexistent")


@pytest.mark.parametrize("row_idx", range(10))
def test_mean_matches_reference(config, data_table, row_idx):
    """Mean-pooling output matches the reference embedding from the parquet."""

    model_class = getattr(embed, config.model_name)
    lc = data_table["lightcurve"][row_idx]
    time = np.array([obs["mjd"] for obs in lc])
    mag = np.array([obs["mag"] for obs in lc])
    expected = np.array(data_table["embedding_mean"][row_idx])

    model = model_class.from_hf(output="mean", reduction=config.ref_reduction)
    embedding = model(time, mag)

    assert embedding.shape == (1, 1, 1, 256)
    assert np.all(np.isfinite(embedding))
    emb_vec = embedding[0, 0, 0].astype(np.float64)
    ref_vec = expected.astype(np.float64)
    cos_sim = np.dot(emb_vec, ref_vec) / (np.linalg.norm(emb_vec) * np.linalg.norm(ref_vec))
    assert cos_sim > config.cos_sim_threshold, f"cosine similarity {cos_sim:.6f} < {config.cos_sim_threshold}"
