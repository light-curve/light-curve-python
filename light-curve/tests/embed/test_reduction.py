"""Unit tests for light_curve.embed reduction strategies (no ONNX session needed)."""

import numpy as np


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
    """preprocess_lc pads/clips to seq_size and returns boolean mask."""
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
