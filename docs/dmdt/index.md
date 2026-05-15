# dm-dt maps

A dm-dt map is a 2D histogram of magnitude differences (dm) versus log-time differences
(lg dt) for all pairs of observations in a light curve.
It was introduced as an input representation for ML classifiers by
[Mahabal et al. 2011](https://ui.adsabs.harvard.edu/abs/2011BASI...39..387M/abstract) and
[Soraisam et al. 2020](https://ui.adsabs.harvard.edu/abs/2020ApJ...892..112S/abstract).

Unlike scalar feature extractors, `DmDt` produces a 2D array — a fixed-size image that
can be fed directly into a CNN.

## Basic usage

```python
import light_curve as lc
import numpy as np

dmdt = lc.DmDt.from_borders(
    min_lgdt=0,       # log10(dt_min) in days
    max_lgdt=2,       # log10(dt_max) in days
    max_abs_dm=3,     # maximum |dm| in magnitudes
    lgdt_size=32,     # number of dt bins
    dm_size=32,       # number of dm bins
    norm=["lgdt", "dm"],
)

t = np.sort(np.random.default_rng(0).uniform(0, 100, 200))
m = 15.0 + 0.1 * np.random.default_rng(0).normal(size=200)

map_ = dmdt.points(t, m)   # shape (32, 32)
```

The output array has shape `(lgdt_size, dm_size)`.
Each cell counts the number of observation pairs \((i, j)\) with \(i < j\) that fall into
the corresponding \((\lg\Delta t,\, \Delta m)\) bin.

## Normalisation

The `norm` parameter controls normalisation of the map:

| `norm` value | Effect |
|---|---|
| `[]` (empty) | Raw counts |
| `["lgdt"]` | Each dt row sums to 1 |
| `["dm"]` | Each dm column sums to 1 |
| `["lgdt", "dm"]` | Normalise by both axes jointly |
| `["nobs"]` | Divide by the number of observations |

## Batch processing

```python
light_curves = [(t1, m1), (t2, m2), ...]
maps = dmdt.many(light_curves)   # shape (N, 32, 32)
```

See the [API reference](api.md) for the full `DmDt` signature including `gaussianise` and
error-weighted variants.
