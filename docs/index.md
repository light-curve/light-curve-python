---
hide:
  - toc
  - navigation
---

# light-curve

**High-performance time-series feature extraction for astrophysics.**

`light-curve` is a Python/Rust library for extracting features from photometric light curves —
fast enough for millions of objects, flexible enough for survey-scale ML pipelines.

```sh title="Install"
pip install 'light-curve[full]'
```

---

<div class="lc-cards">

<div class="lc-card">
<div class="lc-card-svg-wrap">
<svg class="lc-svg-features" viewBox="0 0 240 110" fill="none" xmlns="http://www.w3.org/2000/svg">
  <!-- Light curve: sinusoidal, line goes exactly through all data points -->
  <polyline points="8,60 18,74 28,82 38,79 48,68 58,53 68,41 78,38 88,46 98,60 108,74 118,82 128,79 138,68 148,53 158,41 168,38 178,46"
            stroke="white" stroke-width="1.8" fill="none" opacity="0.9"/>
  <circle cx="8"   cy="60" r="2.5" fill="white" opacity="0.9"/>
  <circle cx="18"  cy="74" r="2.5" fill="white" opacity="0.9"/>
  <circle cx="28"  cy="82" r="2.5" fill="white" opacity="0.9"/>
  <circle cx="38"  cy="79" r="2.5" fill="white" opacity="0.9"/>
  <circle cx="48"  cy="68" r="2.5" fill="white" opacity="0.9"/>
  <circle cx="58"  cy="53" r="2.5" fill="white" opacity="0.9"/>
  <circle cx="68"  cy="41" r="2.5" fill="white" opacity="0.9"/>
  <circle cx="78"  cy="38" r="2.5" fill="white" opacity="0.9"/>
  <circle cx="88"  cy="46" r="2.5" fill="white" opacity="0.9"/>
  <circle cx="98"  cy="60" r="2.5" fill="white" opacity="0.9"/>
  <circle cx="108" cy="74" r="2.5" fill="white" opacity="0.9"/>
  <circle cx="118" cy="82" r="2.5" fill="white" opacity="0.9"/>
  <circle cx="128" cy="79" r="2.5" fill="white" opacity="0.9"/>
  <circle cx="138" cy="68" r="2.5" fill="white" opacity="0.9"/>
  <circle cx="148" cy="53" r="2.5" fill="white" opacity="0.9"/>
  <circle cx="158" cy="41" r="2.5" fill="white" opacity="0.9"/>
  <circle cx="168" cy="38" r="2.5" fill="white" opacity="0.9"/>
  <circle cx="178" cy="46" r="2.5" fill="white" opacity="0.9"/>
  <!-- Annotations: mean line, A bracket, P bracket — fade in on hover -->
  <g class="lc-annotation">
  <!-- Mean line -->
  <line x1="8" y1="60" x2="178" y2="60" stroke="white" stroke-width="1" stroke-dasharray="4,3" opacity="0.5"/>
  <!-- Half-amplitude A: vertical bracket from mean (y=60) to peak (y=38) -->
  <line x1="189" y1="38" x2="189" y2="60" stroke="#ffcc80" stroke-width="1.5"/>
  <line x1="186" y1="38" x2="192" y2="38" stroke="#ffcc80" stroke-width="1.5"/>
  <line x1="186" y1="60" x2="192" y2="60" stroke="#ffcc80" stroke-width="1.5"/>
  <text x="198" y="52" fill="#ffcc80" font-size="10" font-family="monospace" font-weight="bold">A</text>
  <!-- Period P: horizontal bracket between consecutive bright peaks (peak-to-peak) -->
  <line x1="78" y1="28" x2="168" y2="28" stroke="#80deea" stroke-width="1.2"/>
  <line x1="78"  y1="24" x2="78"  y2="32" stroke="#80deea" stroke-width="1.2"/>
  <line x1="168" y1="24" x2="168" y2="32" stroke="#80deea" stroke-width="1.2"/>
  <text x="123" y="20" fill="#80deea" font-size="9" font-family="monospace" font-weight="bold" text-anchor="middle">P</text>
  </g>
</svg>
</div>
<div class="lc-card-body">
<h3><a href="features/">Feature extractors</a></h3>
<p>40+ features across 6 categories: statistical, variability &amp; trend, time sampling, Lomb–Scargle periodogram, parametric fits (Bazin, Villar), and multiband. All implemented in Rust for survey-scale throughput.</p>
</div>
</div>

<div class="lc-card">
<div class="lc-card-svg-wrap">
<svg class="lc-svg-embed" viewBox="0 0 240 110" fill="none" xmlns="http://www.w3.org/2000/svg">
  <!-- Light curve: irregular shape, line goes through points -->
  <polyline points="6,55 12,44 18,36 24,45 30,65 36,72 42,60 48,42 52,55"
            stroke="white" stroke-width="1.5" fill="none" opacity="0.8"/>
  <circle cx="6"  cy="55" r="2.2" fill="white" opacity="0.85"/>
  <circle cx="12" cy="44" r="2.2" fill="white" opacity="0.85"/>
  <circle cx="18" cy="36" r="2.2" fill="white" opacity="0.85"/>
  <circle cx="24" cy="45" r="2.2" fill="white" opacity="0.85"/>
  <circle cx="30" cy="65" r="2.2" fill="white" opacity="0.85"/>
  <circle cx="36" cy="72" r="2.2" fill="white" opacity="0.85"/>
  <circle cx="42" cy="60" r="2.2" fill="white" opacity="0.85"/>
  <circle cx="48" cy="42" r="2.2" fill="white" opacity="0.85"/>
  <circle cx="52" cy="55" r="2.2" fill="white" opacity="0.85"/>
  <!-- Arrow: LC → network -->
  <line x1="57" y1="55" x2="66" y2="55" stroke="white" stroke-width="1.1" opacity="0.5"/>
  <polygon points="66,52 66,58 70,55" fill="white" opacity="0.5"/>
  <!-- Layer 1: input nodes -->
  <circle cx="74" cy="28" r="5.5" fill="white" opacity="0.6"/>
  <circle cx="74" cy="55" r="5.5" fill="white" opacity="0.75"/>
  <circle cx="74" cy="82" r="5.5" fill="white" opacity="0.6"/>
  <!-- Connections L1→L2 (looping animation) -->
  <line class="lc-conn-a" x1="80" y1="28" x2="114" y2="20" stroke="white" stroke-width="1.1"/>
  <line class="lc-conn-a" x1="80" y1="28" x2="114" y2="44" stroke="white" stroke-width="1.1"/>
  <line class="lc-conn-a" x1="80" y1="55" x2="114" y2="44" stroke="white" stroke-width="1.1"/>
  <line class="lc-conn-a" x1="80" y1="55" x2="114" y2="68" stroke="white" stroke-width="1.1"/>
  <line class="lc-conn-a" x1="80" y1="82" x2="114" y2="68" stroke="white" stroke-width="1.1"/>
  <line class="lc-conn-a" x1="80" y1="82" x2="114" y2="90" stroke="white" stroke-width="1.1"/>
  <!-- Layer 2: hidden nodes -->
  <circle cx="118" cy="20" r="4.5" fill="white" opacity="0.55"/>
  <circle cx="118" cy="44" r="4.5" fill="white" opacity="0.7"/>
  <circle cx="118" cy="68" r="4.5" fill="white" opacity="0.7"/>
  <circle cx="118" cy="90" r="4.5" fill="white" opacity="0.55"/>
  <!-- Connections L2→output (looping animation, delayed) -->
  <line class="lc-conn-b" x1="123" y1="20" x2="149" y2="55" stroke="white" stroke-width="1.1"/>
  <line class="lc-conn-b" x1="123" y1="44" x2="149" y2="55" stroke="white" stroke-width="1.1"/>
  <line class="lc-conn-b" x1="123" y1="68" x2="149" y2="55" stroke="white" stroke-width="1.1"/>
  <line class="lc-conn-b" x1="123" y1="90" x2="149" y2="55" stroke="white" stroke-width="1.1"/>
  <!-- Output node (embedding) -->
  <circle cx="155" cy="55" r="6.5" fill="white" opacity="0.85"/>
  <!-- Output vector: appears on hover -->
  <g class="lc-vec-text">
  <text x="165" y="38" fill="#ffcc80" font-size="9" font-family="monospace">[0.31,</text>
  <text x="165" y="51" fill="#ffcc80" font-size="9" font-family="monospace">-1.20,</text>
  <text x="165" y="64" fill="#ffcc80" font-size="9" font-family="monospace"> 0.84,</text>
  <text x="165" y="77" fill="#ffcc80" font-size="9" font-family="monospace"> ...]</text>
  </g>
</svg>
</div>
<div class="lc-card-body">
<h3><a href="embed/">ML embeddings</a></h3>
<p>Map raw light curves to dense vectors using pretrained transformer models (Astromer2, ATCAT). Suitable for classification, anomaly detection, and similarity search at scale.</p>
</div>
</div>

<div class="lc-card">
<div class="lc-card-svg-wrap">
<svg class="lc-svg-dmdt" viewBox="0 0 240 110" fill="none" xmlns="http://www.w3.org/2000/svg">
  <!-- Light curve: 5 points, connected -->
  <polyline points="12,48 27,72 42,55 57,38 72,62" stroke="white" stroke-width="1.5" fill="none" opacity="0.7"/>
  <circle cx="12" cy="48" r="3" fill="white" opacity="0.7"/>
  <circle cx="27" cy="72" r="3" fill="white" opacity="0.7"/>
  <circle cx="42" cy="55" r="3" fill="white" opacity="0.7"/>
  <circle cx="57" cy="38" r="3" fill="white" opacity="0.7"/>
  <circle cx="72" cy="62" r="3" fill="white" opacity="0.7"/>
  <!-- Arrow LC → grid -->
  <line x1="79" y1="55" x2="88" y2="55" stroke="white" stroke-width="1.2" opacity="0.5"/>
  <polygon points="88,52 88,58 92,55" fill="white" opacity="0.5"/>
  <!-- dm-dt grid background (5 cols × 4 rows, 16×16 cells) -->
  <!-- Row 4 (large -Δm, bottom): y=68 -->
  <rect x="96"  y="68" width="16" height="16" fill="white" opacity="0.03" rx="1"/>
  <rect x="114" y="68" width="16" height="16" fill="white" opacity="0.08" rx="1"/>
  <rect x="132" y="68" width="16" height="16" fill="white" opacity="0.15" rx="1"/>
  <rect x="150" y="68" width="16" height="16" fill="white" opacity="0.30" rx="1"/>
  <rect x="168" y="68" width="16" height="16" fill="white" opacity="0.42" rx="1"/>
  <!-- Row 3 (small -Δm): y=50 -->
  <rect x="96"  y="50" width="16" height="16" fill="white" opacity="0.08" rx="1"/>
  <rect x="114" y="50" width="16" height="16" fill="white" opacity="0.28" rx="1"/>
  <rect x="132" y="50" width="16" height="16" fill="white" opacity="0.50" rx="1"/>
  <rect x="150" y="50" width="16" height="16" fill="white" opacity="0.55" rx="1"/>
  <rect x="168" y="50" width="16" height="16" fill="white" opacity="0.22" rx="1"/>
  <!-- Row 2 (small +Δm): y=32 -->
  <rect x="96"  y="32" width="16" height="16" fill="white" opacity="0.10" rx="1"/>
  <rect x="114" y="32" width="16" height="16" fill="white" opacity="0.40" rx="1"/>
  <rect x="132" y="32" width="16" height="16" fill="white" opacity="0.58" rx="1"/>
  <rect x="150" y="32" width="16" height="16" fill="white" opacity="0.30" rx="1"/>
  <rect x="168" y="32" width="16" height="16" fill="white" opacity="0.10" rx="1"/>
  <!-- Row 1 (large +Δm, top): y=14 -->
  <rect x="96"  y="14" width="16" height="16" fill="white" opacity="0.05" rx="1"/>
  <rect x="114" y="14" width="16" height="16" fill="white" opacity="0.15" rx="1"/>
  <rect x="132" y="14" width="16" height="16" fill="white" opacity="0.10" rx="1"/>
  <rect x="150" y="14" width="16" height="16" fill="white" opacity="0.04" rx="1"/>
  <rect x="168" y="14" width="16" height="16" fill="white" opacity="0.02" rx="1"/>
  <!-- Axis label -->
  <text x="137" y="95" fill="white" font-size="8" opacity="0.6" text-anchor="middle">lg Δt →</text>
  <!-- Step 1 (orange): pair p1(12,48)–p2(27,72) → cell col0,row0(large+Δm) at (96,14) -->
  <g class="lc-dmdt-s1">
    <line x1="12" y1="48" x2="27" y2="72" stroke="#ff8f00" stroke-width="2.5"/>
    <circle cx="12" cy="48" r="4" fill="#ff8f00"/>
    <circle cx="27" cy="72" r="4" fill="#ff8f00"/>
    <rect x="96" y="14" width="16" height="16" fill="#ff8f00" rx="1"/>
  </g>
  <!-- Step 2 (cyan): pair p1(12,48)–p4(57,38) → cell col2,row2(small+Δm) at (132,32) -->
  <g class="lc-dmdt-s2">
    <line x1="12" y1="48" x2="57" y2="38" stroke="#00acc1" stroke-width="2.5"/>
    <circle cx="12" cy="48" r="4" fill="#00acc1"/>
    <circle cx="57" cy="38" r="4" fill="#00acc1"/>
    <rect x="132" y="32" width="16" height="16" fill="#00acc1" rx="1"/>
  </g>
  <!-- Step 3 (magenta): pair p3(42,55)–p5(72,62) → cell col1,row3(small-Δm) at (114,50) -->
  <g class="lc-dmdt-s3">
    <line x1="42" y1="55" x2="72" y2="62" stroke="#e040fb" stroke-width="2.5"/>
    <circle cx="42" cy="55" r="4" fill="#e040fb"/>
    <circle cx="72" cy="62" r="4" fill="#e040fb"/>
    <rect x="114" y="50" width="16" height="16" fill="#e040fb" rx="1"/>
  </g>
</svg>
</div>
<div class="lc-card-body">
<h3><a href="dmdt/">dm-dt maps</a></h3>
<p>2D histograms of Δmag vs log-Δt for all observation pairs — fixed-size image representation for CNN-based variability classifiers.</p>
</div>
</div>

</div>

## Quick start

```python
import light_curve as licu
import numpy as np

rng = np.random.default_rng(0)
t   = np.sort(rng.uniform(0, 100, 100))
m   = 15.0 + 0.01 * t + rng.normal(0, 0.1, 100)
err = np.full(100, 0.1)

ext = licu.Extractor(licu.Amplitude(), licu.BeyondNStd(nstd=1), licu.LinearFit())
result = ext(t, m, err)
print(dict(zip(ext.names, result)))
# {'amplitude': 0.67, 'beyond_1_std': 0.35, 'linear_fit_slope': 0.010, ...}
```

Use `.many()` for batch processing of many light curves with reduced Python–Rust overhead:

```python
light_curves = [(t1, m1, err1), (t2, m2, err2), ...]
amplitudes = licu.Amplitude().many(light_curves)   # shape (N,)
```
