# light-curve

**High-performance time-series feature extraction for astrophysics.**

`light-curve` is a Python/Rust library for extracting features from photometric light curves —
fast enough for millions of objects, flexible enough for survey-scale ML pipelines.

```sh title="Install"
pip install 'light-curve[full]'
```

---

<div class="lc-cards" markdown>

<a href="features/" class="lc-card">
<div class="lc-card-svg-wrap">
<svg class="lc-svg-features" viewBox="0 0 240 110" fill="none" xmlns="http://www.w3.org/2000/svg">
  <!-- light curve -->
  <polyline points="8,78 22,58 36,82 50,52 64,72 78,38 92,62 106,78 120,46 134,68 148,82 162,54 176,72 190,62 204,76 218,84" stroke="white" stroke-width="1.8" fill="none" opacity="0.85"/>
  <!-- scatter dots -->
  <circle cx="8"   cy="80"  r="2.8" fill="white" opacity="0.9"/>
  <circle cx="22"  cy="60"  r="2.8" fill="white" opacity="0.9"/>
  <circle cx="36"  cy="84"  r="2.8" fill="white" opacity="0.9"/>
  <circle cx="50"  cy="54"  r="2.8" fill="white" opacity="0.9"/>
  <circle cx="64"  cy="74"  r="2.8" fill="white" opacity="0.9"/>
  <circle cx="78"  cy="40"  r="2.8" fill="white" opacity="0.9"/>
  <circle cx="92"  cy="64"  r="2.8" fill="white" opacity="0.9"/>
  <circle cx="106" cy="80"  r="2.8" fill="white" opacity="0.9"/>
  <circle cx="120" cy="48"  r="2.8" fill="white" opacity="0.9"/>
  <circle cx="134" cy="70"  r="2.8" fill="white" opacity="0.9"/>
  <circle cx="148" cy="84"  r="2.8" fill="white" opacity="0.9"/>
  <circle cx="162" cy="56"  r="2.8" fill="white" opacity="0.9"/>
  <circle cx="176" cy="74"  r="2.8" fill="white" opacity="0.9"/>
  <circle cx="190" cy="64"  r="2.8" fill="white" opacity="0.9"/>
  <circle cx="204" cy="78"  r="2.8" fill="white" opacity="0.9"/>
  <circle cx="218" cy="86"  r="2.8" fill="white" opacity="0.9"/>
  <!-- amplitude bracket — appears on hover -->
  <g class="lc-amplitude">
    <line x1="232" y1="38" x2="232" y2="88" stroke="#ffcc80" stroke-width="2"/>
    <line x1="228" y1="38" x2="236" y2="38" stroke="#ffcc80" stroke-width="2"/>
    <line x1="228" y1="88" x2="236" y2="88" stroke="#ffcc80" stroke-width="2"/>
    <text x="195" y="68" fill="#ffcc80" font-size="11" font-family="monospace" font-weight="bold">A</text>
    <!-- trend line -->
    <line x1="8" y1="75" x2="218" y2="68" stroke="#ffcc80" stroke-width="1.5" stroke-dasharray="5,3" opacity="0.8"/>
  </g>
</svg>
</div>
<div class="lc-card-body">
<h3>Hand-crafted features</h3>
<p>40+ extractors: amplitude, period, variability indices, parametric fits. Shared interface, batch processing, equations in every docstring.</p>
</div>
</a>

<a href="embed/" class="lc-card">
<div class="lc-card-svg-wrap">
<svg class="lc-svg-embed" viewBox="0 0 240 110" fill="none" xmlns="http://www.w3.org/2000/svg">
  <!-- input: mini light curve -->
  <polyline points="6,75 14,58 22,80 30,50 38,70 46,40 54,62" stroke="white" stroke-width="1.5" fill="none" opacity="0.7"/>
  <!-- input nodes -->
  <circle cx="68" cy="30"  r="6" fill="white" opacity="0.55"/>
  <circle cx="68" cy="55"  r="6" fill="white" opacity="0.75"/>
  <circle cx="68" cy="80"  r="6" fill="white" opacity="0.55"/>
  <!-- hidden layer nodes -->
  <circle cx="120" cy="20"  r="6" fill="white" opacity="0.45"/>
  <circle cx="120" cy="42"  r="6" fill="white" opacity="0.7"/>
  <circle cx="120" cy="62"  r="6" fill="white" opacity="0.7"/>
  <circle cx="120" cy="82"  r="6" fill="white" opacity="0.45"/>
  <circle cx="120" cy="100" r="6" fill="white" opacity="0.35"/>
  <!-- output node -->
  <circle cx="172" cy="55"  r="8" fill="white" opacity="0.85"/>
  <!-- animated connections input→hidden -->
  <line class="lc-conn" x1="74" y1="55" x2="114" y2="42" stroke="white" stroke-width="1.2" opacity="0.5"/>
  <line class="lc-conn" x1="74" y1="55" x2="114" y2="62" stroke="white" stroke-width="1.2" opacity="0.5"/>
  <line class="lc-conn" x1="74" y1="30" x2="114" y2="20" stroke="white" stroke-width="1.2" opacity="0.4"/>
  <line class="lc-conn" x1="74" y1="80" x2="114" y2="82" stroke="white" stroke-width="1.2" opacity="0.4"/>
  <!-- animated connections hidden→output -->
  <line class="lc-conn" x1="126" y1="42" x2="164" y2="55" stroke="white" stroke-width="1.2" opacity="0.5"/>
  <line class="lc-conn" x1="126" y1="62" x2="164" y2="55" stroke="white" stroke-width="1.2" opacity="0.5"/>
  <!-- output vector label — appears on hover -->
  <text class="lc-vec-text" x="184" y="40" fill="#ffcc80" font-size="9" font-family="monospace">[0.31,</text>
  <text class="lc-vec-text" x="184" y="53" fill="#ffcc80" font-size="9" font-family="monospace">-1.20,</text>
  <text class="lc-vec-text" x="184" y="66" fill="#ffcc80" font-size="9" font-family="monospace"> 0.84,</text>
  <text class="lc-vec-text" x="184" y="79" fill="#ffcc80" font-size="9" font-family="monospace"> ...]</text>
</svg>
</div>
<div class="lc-card-body">
<h3>ML embeddings</h3>
<p>Embed light curves with pretrained ONNX models — Astromer2 (256-dim, single-band) and ATCAT (384-dim, 6-band LSST). One call to embed thousands.</p>
</div>
</a>

<a href="dmdt/" class="lc-card">
<div class="lc-card-svg-wrap">
<svg class="lc-svg-dmdt" viewBox="0 0 240 110" fill="none" xmlns="http://www.w3.org/2000/svg">
  <!-- axis labels -->
  <text x="6" y="106" fill="white" font-size="9" opacity="0.7">lg Δt</text>
  <text x="2" y="14"  fill="white" font-size="9" opacity="0.7" transform="rotate(-90,10,55)">Δm</text>
  <!-- static cells: 6×5 grid, some filled -->
  <rect x="28" y="72" width="18" height="18" fill="white" opacity="0.12" rx="2"/>
  <rect x="48" y="72" width="18" height="18" fill="white" opacity="0.22" rx="2"/>
  <rect x="68" y="72" width="18" height="18" fill="white" opacity="0.35" rx="2"/>
  <rect x="88" y="72" width="18" height="18" fill="white" opacity="0.20" rx="2"/>
  <rect x="108" y="72" width="18" height="18" fill="white" opacity="0.10" rx="2"/>
  <rect x="128" y="72" width="18" height="18" fill="white" opacity="0.08" rx="2"/>

  <rect x="28" y="52" width="18" height="18" fill="white" opacity="0.18" rx="2"/>
  <rect x="48" y="52" width="18" height="18" fill="white" opacity="0.45" rx="2"/>
  <rect x="68" y="52" width="18" height="18" fill="white" opacity="0.70" rx="2"/>
  <rect x="88" y="52" width="18" height="18" fill="white" opacity="0.55" rx="2"/>
  <rect x="108" y="52" width="18" height="18" fill="white" opacity="0.25" rx="2"/>
  <rect x="128" y="52" width="18" height="18" fill="white" opacity="0.10" rx="2"/>

  <rect x="28" y="32" width="18" height="18" fill="white" opacity="0.08" rx="2"/>
  <rect x="48" y="32" width="18" height="18" fill="white" opacity="0.20" rx="2"/>
  <rect x="68" y="32" width="18" height="18" fill="white" opacity="0.40" rx="2"/>
  <rect x="88" y="32" width="18" height="18" fill="white" opacity="0.60" rx="2"/>
  <rect x="108" y="32" width="18" height="18" fill="white" opacity="0.35" rx="2"/>
  <rect x="128" y="32" width="18" height="18" fill="white" opacity="0.15" rx="2"/>

  <rect x="28" y="12" width="18" height="18" fill="white" opacity="0.05" rx="2"/>
  <rect x="48" y="12" width="18" height="18" fill="white" opacity="0.10" rx="2"/>
  <rect x="68" y="12" width="18" height="18" fill="white" opacity="0.15" rx="2"/>
  <rect x="88" y="12" width="18" height="18" fill="white" opacity="0.25" rx="2"/>
  <rect x="108" y="12" width="18" height="18" fill="white" opacity="0.45" rx="2"/>
  <rect x="128" y="12" width="18" height="18" fill="white" opacity="0.30" rx="2"/>

  <!-- hover: extra columns appear with staggered delay -->
  <rect class="lc-cell-hover" x="148" y="72" width="18" height="18" fill="white" opacity="0.06" rx="2"/>
  <rect class="lc-cell-hover" x="168" y="72" width="18" height="18" fill="white" opacity="0.04" rx="2"/>
  <rect class="lc-cell-hover" x="148" y="52" width="18" height="18" fill="white" opacity="0.08" rx="2"/>
  <rect class="lc-cell-hover" x="168" y="52" width="18" height="18" fill="white" opacity="0.06" rx="2"/>
  <rect class="lc-cell-hover" x="148" y="32" width="18" height="18" fill="white" opacity="0.20" rx="2"/>
  <rect class="lc-cell-hover" x="168" y="32" width="18" height="18" fill="white" opacity="0.35" rx="2"/>
  <rect class="lc-cell-hover" x="148" y="12" width="18" height="18" fill="white" opacity="0.55" rx="2"/>
  <rect class="lc-cell-hover" x="168" y="12" width="18" height="18" fill="white" opacity="0.70" rx="2"/>
</svg>
</div>
<div class="lc-card-body">
<h3>dm-dt maps</h3>
<p>2D histograms of Δmag vs log-Δt for all observation pairs — a natural CNN input for variability classification.</p>
</div>
</a>

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
