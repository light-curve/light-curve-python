import arro3.core
import nanoarrow
import numpy as np
import polars as pl
import pyarrow as pa
import pytest
from light_curve_ext.test_feature import gen_feature_evaluators, gen_lc
from numpy.testing import assert_array_equal

from light_curve import light_curve_ext as lc


def _make_arrow_lcs(lcs, dtype=np.float64):
    """Build a pyarrow List<Struct<t, m, sigma>> array from list of (t, m, sigma) tuples."""
    pa_type = pa.float32() if dtype == np.float32 else pa.float64()
    struct_type = pa.struct([("t", pa_type), ("m", pa_type), ("sigma", pa_type)])
    # Build each light curve as a list of struct rows
    all_lcs = []
    for t, m, sigma in lcs:
        rows = [{"t": dtype(ti), "m": dtype(mi), "sigma": dtype(si)} for ti, mi, si in zip(t, m, sigma)]
        all_lcs.append(rows)
    return pa.array(all_lcs, type=pa.list_(struct_type))


def _make_arrow_lcs_no_sigma(lcs, dtype=np.float64):
    """Build a pyarrow List<Struct<t, m>> array from list of (t, m, sigma) tuples (sigma ignored)."""
    pa_type = pa.float32() if dtype == np.float32 else pa.float64()
    struct_type = pa.struct([("t", pa_type), ("m", pa_type)])
    all_lcs = []
    for t, m, _sigma in lcs:
        rows = [{"t": dtype(ti), "m": dtype(mi)} for ti, mi in zip(t, m)]
        all_lcs.append(rows)
    return pa.array(all_lcs, type=pa.list_(struct_type))


@pytest.mark.parametrize("feature", gen_feature_evaluators(parametric_variants=0))
def test_many_arrow_3fields(feature):
    """Arrow List<Struct<f64, f64, f64>> matches list-of-tuples result."""
    rng = np.random.default_rng(42)
    n_obs, n_lc = 64, 8
    lcs = [gen_lc(n_obs, rng=rng) for _ in range(n_lc)]

    expected = feature.many(lcs, sorted=True, n_jobs=1)
    arrow_arr = _make_arrow_lcs(lcs)
    result = feature.many(arrow_arr, sorted=True, n_jobs=1)
    assert_array_equal(expected, result)


@pytest.mark.parametrize("feature", gen_feature_evaluators(parametric_variants=0))
def test_many_arrow_2fields(feature):
    """Arrow List<Struct<f64, f64>> (no sigma) matches list-of-tuples result."""
    rng = np.random.default_rng(42)
    n_obs, n_lc = 64, 8
    lcs = [gen_lc(n_obs, rng=rng) for _ in range(n_lc)]

    lcs_no_sigma = [(t, m, None) for t, m, _s in lcs]
    expected = feature.many(lcs_no_sigma, sorted=True, n_jobs=1)
    arrow_arr = _make_arrow_lcs_no_sigma(lcs)
    result = feature.many(arrow_arr, sorted=True, n_jobs=1)
    assert_array_equal(expected, result)


def test_many_arrow_f32():
    """Arrow Float32 variant works correctly."""
    rng = np.random.default_rng(42)
    n_obs, n_lc = 64, 8
    lcs = [gen_lc(n_obs, rng=rng) for _ in range(n_lc)]

    feature = lc.Amplitude()
    lcs_f32 = [(t.astype(np.float32), m.astype(np.float32), s.astype(np.float32)) for t, m, s in lcs]
    expected = feature.many(lcs_f32, sorted=True, n_jobs=1)
    arrow_arr = _make_arrow_lcs(lcs, dtype=np.float32)
    result = feature.many(arrow_arr, sorted=True, n_jobs=1)
    assert_array_equal(expected, result)
    assert result.dtype == np.float32


def test_many_arrow_chunked():
    """Arrow ChunkedArray with multiple chunks works."""
    rng = np.random.default_rng(42)
    n_obs, n_lc = 64, 8
    lcs = [gen_lc(n_obs, rng=rng) for _ in range(n_lc)]

    feature = lc.Amplitude()
    expected = feature.many(lcs, sorted=True, n_jobs=1)

    # Split into two chunks
    mid = n_lc // 2
    chunk1 = _make_arrow_lcs(lcs[:mid])
    chunk2 = _make_arrow_lcs(lcs[mid:])
    chunked = pa.chunked_array([chunk1, chunk2])
    result = feature.many(chunked, sorted=True, n_jobs=1)
    assert_array_equal(expected, result)


def test_many_arrow_wrong_fields():
    """Arrow input with wrong schema raises appropriate errors."""
    feature = lc.Amplitude()

    # Wrong number of struct fields (1 field)
    arr = pa.array(
        [[{"x": 1.0}], [{"x": 2.0}]],
        type=pa.list_(pa.struct([("x", pa.float64())])),
    )
    with pytest.raises(ValueError, match="2 .* or 3"):
        feature.many(arr, sorted=True)

    # Wrong field dtype (int32)
    arr = pa.array(
        [[{"t": 1, "m": 2}]],
        type=pa.list_(pa.struct([("t", pa.int32()), ("m", pa.int32())])),
    )
    with pytest.raises(TypeError, match="Float32 or Float64"):
        feature.many(arr, sorted=True)

    # Non-list type
    arr = pa.array([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="List array"):
        feature.many(arr, sorted=True)

    # Mixed field dtypes
    arr = pa.array(
        [[{"t": np.float32(1.0), "m": np.float64(2.0)}]],
        type=pa.list_(pa.struct([("t", pa.float32()), ("m", pa.float64())])),
    )
    with pytest.raises(TypeError, match="same dtype"):
        feature.many(arr, sorted=True)


def test_many_arrow_parallel():
    """Arrow input works with parallel execution."""
    rng = np.random.default_rng(42)
    n_obs, n_lc = 64, 16
    lcs = [gen_lc(n_obs, rng=rng) for _ in range(n_lc)]

    feature = lc.Amplitude()
    expected = feature.many(lcs, sorted=True, n_jobs=1)
    arrow_arr = _make_arrow_lcs(lcs)
    result = feature.many(arrow_arr, sorted=True, n_jobs=2)
    assert_array_equal(expected, result)


def test_many_polars():
    """Polars Series (Arrow-backed) works with many()."""
    rng = np.random.default_rng(42)
    n_obs, n_lc = 64, 8
    lcs = [gen_lc(n_obs, rng=rng) for _ in range(n_lc)]

    feature = lc.Amplitude()
    expected = feature.many(lcs, sorted=True, n_jobs=1)

    # Build a Polars Series of list-of-structs
    arrow_arr = _make_arrow_lcs(lcs)
    polars_series = pl.Series(arrow_arr)
    result = feature.many(polars_series, sorted=True, n_jobs=1)
    assert_array_equal(expected, result)


def test_many_nanoarrow():
    """nanoarrow array works with many()."""
    rng = np.random.default_rng(42)
    n_obs, n_lc = 64, 8
    lcs = [gen_lc(n_obs, rng=rng) for _ in range(n_lc)]

    feature = lc.Amplitude()
    expected = feature.many(lcs, sorted=True, n_jobs=1)

    arrow_arr = _make_arrow_lcs(lcs)
    nano_arr = nanoarrow.Array(arrow_arr)
    result = feature.many(nano_arr, sorted=True, n_jobs=1)
    assert_array_equal(expected, result)


def test_many_arro3():
    """arro3 array works with many()."""
    rng = np.random.default_rng(42)
    n_obs, n_lc = 64, 8
    lcs = [gen_lc(n_obs, rng=rng) for _ in range(n_lc)]

    feature = lc.Amplitude()
    expected = feature.many(lcs, sorted=True, n_jobs=1)

    arrow_arr = _make_arrow_lcs(lcs)
    arro3_arr = arro3.core.Array.from_arrow(arrow_arr)
    result = feature.many(arro3_arr, sorted=True, n_jobs=1)
    assert_array_equal(expected, result)
