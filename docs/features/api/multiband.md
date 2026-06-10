# Multiband features API

## Multiband support in standard features

All standard feature extractors accept a `bands=` constructor argument to enable multiband
mode.  Full API documentation is available in the per-category reference pages:

- [Meta features (`Extractor`, `Bins`)](../api/meta.md) — `Extractor` supports mixed
  single-band + multiband, `Bins` supports `bands=` for per-passband binning.
- [Periodogram](../api/periodogram.md) — `Periodogram` supports `bands=` and
  `multiband_normalization=`.
- [Variability](../api/variability.md), [Linear trend](../api/linear.md),
  [Time sampling](../api/time.md), [Non-linear parametric fits](../api/fitting.md) —
  all accept `bands=`.

---

## Pure multiband features

These features are inherently multiband — they always require `bands` and have no single-band mode.

::: light_curve.ColorOfMaximum

---

::: light_curve.ColorOfMedian

---

::: light_curve.ColorOfMinimum

---

::: light_curve.ColorSpread
