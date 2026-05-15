# light-curve

**High-performance time-series feature extraction for astrophysics.**

`light-curve` is a Python/Rust library for extracting features from photometric light curves —
fast enough for millions of objects, flexible enough for survey-scale ML pipelines.

<div class="grid" markdown>

```sh title="Install"
pip install 'light-curve[full]'
```

</div>

---

<div class="grid cards" markdown>

-   :material-chart-line: **Hand-crafted features**

    ---

    40+ feature extractors: amplitude, period, variability indices, parametric fits, and more.
    Shared interface, batch processing, LaTeX equations in every docstring.

    [:octicons-arrow-right-24: Features](features/index.md)

-   :material-brain: **ML embeddings**

    ---

    Embed light curves with pretrained ONNX models — Astromer2 (256-dim) and ATCAT (384-dim).
    One line to download weights, one line to embed.

    [:octicons-arrow-right-24: Embeddings](embed/index.md)

-   :material-grid: **dm-dt maps**

    ---

    2D histograms of magnitude difference vs log-time difference — a natural CNN input
    for variability classification.

    [:octicons-arrow-right-24: dm-dt maps](dmdt/index.md)

</div>

## Quick start

```python
import light_curve as lc
import numpy as np

rng = np.random.default_rng(0)
t   = np.sort(rng.uniform(0, 100, 100))
m   = 15.0 + 0.01 * t + rng.normal(0, 0.1, 100)
err = np.full(100, 0.1)

ext = lc.Extractor(lc.Amplitude(), lc.BeyondNStd(nstd=1), lc.LinearFit())
result = ext(t, m, err)
print(dict(zip(ext.names, result)))
# {'amplitude': 0.67, 'beyond_1_std': 0.35, 'linear_fit_slope': 0.010, ...}
```

Use `.many()` for batch processing of many light curves with reduced Python–Rust overhead:

```python
light_curves = [(t1, m1, err1), (t2, m2, err2), ...]
amplitudes = lc.Amplitude().many(light_curves)   # shape (N,)
```

---

## Why light-curve?

| | light-curve |
|---|---|
| **Speed** | Rust core; `.many()` amortises Python–Rust call overhead across the full dataset |
| **Accuracy** | Equations from the literature, documented with references and LaTeX |
| **Flexibility** | 40+ scalar features + ONNX embeddings + dm-dt maps in one package |
| **Compatibility** | PyArrow, Polars, nested-pandas inputs; NumPy arrays out |
| **Portability** | Binary wheels for Linux x86-64/aarch64, macOS, Windows; no Rust toolchain needed |
