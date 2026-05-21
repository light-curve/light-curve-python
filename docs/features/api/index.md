# API Reference

All feature extractor classes share the same calling interface: `__call__` for a single light curve and `many` for a batch.

## Common attributes

Every extractor exposes two read-only attributes:

| Attribute | Type | Description |
|---|---|---|
| `names` | `list[str]` | Output column names, one per extracted value |
| `descriptions` | `list[str]` | Human-readable description of each output |

```python
import light_curve as lc

ext = lc.Extractor(lc.Amplitude(), lc.LinearFit())
print(ext.names)        # ['amplitude', 'linear_fit_slope', ...]
print(ext.descriptions) # ['Half amplitude of magnitude sample', ...]
```

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

## JSON serialization

Every extractor can be serialized to and from JSON, which preserves the class type and all constructor parameters.

### `to_json()` — serialize

```python
s = extractor.to_json()  # returns str
```

### `feature_from_json()` — deserialize

```python
import light_curve as lc

extractor = lc.feature_from_json(s)
```

Returns a `JSONDeserializedFeature` that behaves identically to the original extractor (same `names`, `descriptions`, and `__call__` / `many` interface).

**Round-trip example:**

```python
import light_curve as lc
import numpy as np

original = lc.Extractor(lc.Amplitude(), lc.LinearFit())
s = original.to_json()

restored = lc.feature_from_json(s)
assert restored.names == original.names

t = np.sort(np.random.default_rng(0).uniform(0, 100, 100))
m = np.random.default_rng(1).normal(15, 0.2, 100)
assert np.allclose(original(t, m), restored(t, m))
```

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
| [Multiband](../multiband/api.md) | ColorOfMedian, RainbowFit |
