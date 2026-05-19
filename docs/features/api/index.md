# API Reference

All feature extractor classes share the same calling interface: `__call__` for a single light curve and `many` for a batch.

## Common interface

### `__call__` — single light curve

```python
extractor(t, m, sigma=None, *, fill_value=None, sorted=None, check=True, cast=False)
```

Extract features and return them as a numpy array.

| Parameter | Type | Description |
|---|---|---|
| `t` | `np.ndarray` (float32 or float64) | Time moments |
| `m` | `np.ndarray` | Magnitudes or fluxes |
| `sigma` | `np.ndarray`, optional | Photometric errors; assumed unity if `None` |
| `fill_value` | `float` or `None` | Value for invalid features; raises if `None` |
| `sorted` | `bool` or `None` | Whether `t` is sorted; checked if `None` |
| `check` | `bool` | Validate inputs for NaNs / infinities |
| `cast` | `bool` | Allow non-ndarray input and automatic dtype casting |

**Returns** `np.ndarray` (float32 or float64) — one value per extracted feature.

### `many` — batch processing

```python
extractor.many(lcs, *, fill_value=None, sorted=None, check=True, cast=False, n_jobs=-1)
```

Parallel feature extraction over multiple light curves. Equivalent to:

```python
np.stack([extractor(*lc, fill_value=fill_value, sorted=sorted, check=check)
          for lc in lcs])
```

| Parameter | Type | Description |
|---|---|---|
| `lcs` | list of `(t, m)` or `(t, m, sigma)` tuples, or Arrow array | Input light curves |
| `arrow_fields` | `list[str or int]` | Required for Arrow input: field names/indices for `[t, m]` or `[t, m, sigma]` |
| `fill_value` | `float` or `None` | Same as `__call__` |
| `sorted` | `bool` or `None` | Same as `__call__` |
| `check` | `bool` | Same as `__call__` |
| `n_jobs` | `int` | Parallel workers; `-1` uses all CPU cores |

**Returns** `np.ndarray` of shape `(N, n_features)`.

---

## Feature pages

| Page | Contents |
|---|---|
| [Meta](meta.md) | `Extractor`, `Bins` |
| [Statistical](statistical.md) | Amplitude, BeyondNStd, Cusum, Kurtosis, LinearFit, Mean, Skew, … |
| [Variability & trend](variability.md) | EtaE, ExcessVariance, ReducedChi2, … |
| [Time sampling](time.md) | ObservationCount, TimeMean, TimeStandardDeviation |
| [Periodogram](periodogram.md) | Periodogram |
| [Parametric fits](fitting.md) | BazinFit, VillarFit |
| [Detection-based](detection.md) | Duration, MaximumTimeInterval, … |
| [Multiband](multiband.md) | MultibandFeature |
