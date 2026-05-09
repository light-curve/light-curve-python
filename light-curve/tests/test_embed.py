"""Tests for light_curve.embed — Astromer1 and Astromer2 embedding models.

ONNX models are downloaded from HuggingFace via huggingface_hub (cached in
the HF default cache directory).  Expected embeddings come from the prep-models
submodule at ``tests/prep-models/``.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from huggingface_hub import hf_hub_download

PREP_MODELS_DIR = Path(__file__).parent / "prep-models" / "models"


@dataclass(frozen=True)
class AstromerTestConfig:
    model_name: str
    repo_id: str
    test_parquet: Path
    # reduction used when generating the reference parquet
    ref_reduction: str
    # Astromer1 has larger TF→ONNX drift (~0.98) than Astromer2 (~0.999)
    cos_sim_threshold: float


ASTROMER_CONFIGS = [
    AstromerTestConfig(
        model_name="Astromer1",
        repo_id="light-curve/astromer1",
        test_parquet=PREP_MODELS_DIR / "astromer1" / "out" / "test-data" / "astromer1_test.parquet",
        ref_reduction="beginning",
        cos_sim_threshold=0.97,
    ),
    AstromerTestConfig(
        model_name="Astromer2",
        repo_id="light-curve/astromer2",
        test_parquet=PREP_MODELS_DIR / "astromer2" / "out" / "test-data" / "astromer2_test.parquet",
        ref_reduction="non-overlapping-windows",
        cos_sim_threshold=0.999,
    ),
]


# ---------------------------------------------------------------------------
# Shared fixtures (parametrized over both models)
# ---------------------------------------------------------------------------


@pytest.fixture(
    scope="session",
    params=ASTROMER_CONFIGS,
    ids=[c.model_name.lower() for c in ASTROMER_CONFIGS],
)
def astromer_config(request) -> AstromerTestConfig:
    return request.param


@pytest.fixture(scope="session")
def astromer_session(astromer_config):
    import onnxruntime as ort

    prefix = astromer_config.repo_id.split("/")[-1]
    path = hf_hub_download(repo_id=astromer_config.repo_id, filename=f"{prefix}.onnx")
    return ort.InferenceSession(str(path))


@pytest.fixture(scope="session")
def astromer_test_table(astromer_config):
    return pq.read_table(astromer_config.test_parquet).to_pydict()


# ---------------------------------------------------------------------------
# Preprocessing tests (no ONNX session needed)
# ---------------------------------------------------------------------------


def test_non_overlapping_windows_short_lc():
    """LC shorter than seq_size produces one window."""
    from light_curve.embed import NonOverlappingWindows

    tr = NonOverlappingWindows()
    time = np.arange(50, dtype=float)
    mag = np.ones(50)
    subsamples = tr.subsample_lc(time, mag, seq_size=200)
    assert len(subsamples) == 1
    assert subsamples[0][0].shape == (50,)


def test_non_overlapping_windows_long_lc():
    """LC of length 400 with seq_size=200 gives 2 windows."""
    from light_curve.embed import NonOverlappingWindows

    tr = NonOverlappingWindows()
    time = np.arange(400, dtype=float)
    mag = np.ones(400)
    subsamples = tr.subsample_lc(time, mag, seq_size=200)
    assert len(subsamples) == 2


def test_preprocess_lc_shape():
    """preprocess_lc pads/clips to seq_size and returns mask."""
    from light_curve.embed import NonOverlappingWindows

    tr = NonOverlappingWindows()
    time = np.arange(50, dtype=float)
    mag = np.ones(50)
    time_out, mag_out, mask = tr.preprocess_lc(time, mag, seq_size=200)
    assert time_out.shape == (1, 200)
    assert mag_out.shape == (1, 200)
    assert mask.shape == (1, 200)
    assert mask[0, :50].all()
    assert not mask[0, 50:].any()


@pytest.mark.parametrize(
    "model_class_name",
    ["Astromer1", "Astromer2"],
)
def test_astromer_preprocess_normalises(model_class_name):
    """preprocess_lc produces zero-mean time and mag for each window."""
    import light_curve.embed as lce

    model_class = getattr(lce, model_class_name)
    model = model_class(session=None, reduction="non-overlapping-windows")
    time = np.linspace(0, 100, 200)
    mag = np.linspace(10, 15, 200)
    tensors = model.preprocess_lc(time, mag)
    valid = tensors.bool_mask[0]
    assert abs(tensors.times[0, valid, 0].mean()) < 1e-5
    assert abs(tensors.input[0, valid, 0].mean()) < 1e-5


# ---------------------------------------------------------------------------
# End-to-end tests (parametrized over Astromer1 / Astromer2 via fixture)
# ---------------------------------------------------------------------------


def test_non_overlapping_windows_sequence_output_long_lc(astromer_config, astromer_session):
    """NonOverlappingWindows + 'sequence' output yields consistent (1,1,200,256) shape."""
    import light_curve.embed as lce

    model_class = getattr(lce, astromer_config.model_name)
    model = model_class(session=astromer_session, output="sequence", reduction="non-overlapping-windows")

    # 350 obs → 2 windows: 200 valid + 150 valid (+50 padded)
    time = np.linspace(0, 1000, 350)
    mag = np.ones(350)
    embedding = model(time, mag)

    # shape: (BAND=1, SUBSAMPLE=1, SEQUENCE=200, VALUE=256) — masked mean over windows per timestep
    assert embedding.shape == (1, 1, 200, 256)
    assert np.all(np.isfinite(embedding))


def test_from_hf_shape(astromer_config):
    """from_hf() returns a working model with correct output shape."""
    import light_curve.embed as lce

    model_class = getattr(lce, astromer_config.model_name)
    model = model_class.from_hf(output="mean")
    time = np.linspace(0, 100, 50)
    mag = np.ones(50)
    embedding = model(time, mag)
    assert embedding.shape == (1, 1, 1, 256)
    assert np.all(np.isfinite(embedding))


def test_from_hf_matches_session(astromer_config, astromer_session, astromer_test_table):
    """from_hf() and a manually created session produce identical embeddings."""
    import light_curve.embed as lce

    model_class = getattr(lce, astromer_config.model_name)
    lc = astromer_test_table["lightcurve"][0]
    time = np.array([obs["mjd"] for obs in lc])
    mag = np.array([obs["mag"] for obs in lc])

    model_hf = model_class.from_hf(output="mean")
    model_manual = model_class(session=astromer_session, output="mean")

    emb_hf = model_hf(time, mag)
    emb_manual = model_manual(time, mag)
    np.testing.assert_array_equal(emb_hf, emb_manual)


def test_from_hf_invalid_output(astromer_config):
    """from_hf() raises ValueError for unknown output names."""
    import light_curve.embed as lce

    model_class = getattr(lce, astromer_config.model_name)
    with pytest.raises(ValueError, match="Unknown output"):
        model_class.from_hf(output="nonexistent")


@pytest.mark.parametrize("row_idx", range(10))
def test_mean_matches_reference(astromer_config, astromer_session, astromer_test_table, row_idx):
    """Mean-pooling output matches the reference embedding from the parquet."""
    import light_curve.embed as lce

    model_class = getattr(lce, astromer_config.model_name)
    lc = astromer_test_table["lightcurve"][row_idx]
    time = np.array([obs["mjd"] for obs in lc])
    mag = np.array([obs["mag"] for obs in lc])
    expected = np.array(astromer_test_table["embedding_mean"][row_idx])

    model = model_class(session=astromer_session, output="mean", reduction=astromer_config.ref_reduction)
    embedding = model(time, mag)

    assert embedding.shape == (1, 1, 1, 256)
    assert np.all(np.isfinite(embedding))
    emb_vec = embedding[0, 0, 0].astype(np.float64)
    ref_vec = expected.astype(np.float64)
    cos_sim = np.dot(emb_vec, ref_vec) / (np.linalg.norm(emb_vec) * np.linalg.norm(ref_vec))
    assert cos_sim > astromer_config.cos_sim_threshold, (
        f"cosine similarity {cos_sim:.6f} < {astromer_config.cos_sim_threshold}"
    )
