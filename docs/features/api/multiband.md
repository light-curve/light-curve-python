# Multiband features API

## Rust-based multiband support

All standard Rust feature extractors accept a `bands=` constructor argument to enable multiband
mode.  Full API documentation is available in the per-category reference pages:

- [Meta features (`Extractor`, `Bins`)](../api/meta.md) — `Extractor` supports mixed
  single-band + multiband, `Bins` supports `bands=` for per-passband binning.
- [Periodogram](../api/periodogram.md) — `Periodogram` supports `bands=` and
  `multiband_normalization=`.
- [Statistical](../api/statistical.md), [Variability](../api/variability.md),
  [Time sampling](../api/time.md), [Parametric fits](../api/fitting.md) —
  all accept `bands=`.

---

## Rust-based pure multiband features

These features are inherently multiband — they always require `bands` and have no single-band mode.

::: light_curve.ColorOfMaximum

---

::: light_curve.ColorOfMedian

---

::: light_curve.ColorOfMinimum

---

::: light_curve.ColorSpread

---

## Experimental pure-Python features

::: light_curve.RainbowFit
