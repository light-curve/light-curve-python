---
hide:
  - toc
  - navigation
---

# light-curve

**High-performance time-series feature extraction for astrophysics.**

`light-curve` is a Python package for analyzing photometric light curves at the scale of millions of objects.
It provides multiple tools for ML pre-processing pipelines as well as 40+ statistical and variability features
for filtering, classification, and catalog analysis.

```sh title="Install"
pip install 'light-curve[full]'
```

---

<div class="lc-cards">

<div class="lc-card">
<div class="lc-card-svg-wrap">
<svg class="lc-svg-features" viewBox="0 0 240 110" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path stroke="#FFFFFF" stroke-width="2" d="M9.902,53.198C16.77,67.04,23.081,82.016,36.736,82.016c25.387,0,25.387-51.762,50.776-51.762c25.386,0,25.386,51.762,50.774,51.762c25.391,0,25.391-51.762,50.776-51.762c8.896,0,14.674,6.355,19.521,14.61"/>
  <circle fill="#FFFFFF" cx="75.023"  cy="35.664" r="2.986"/>
  <circle fill="#FFFFFF" cx="69.051"  cy="43.115" r="2.986"/>
  <circle fill="#FFFFFF" cx="61.555"  cy="57.213" r="2.986"/>
  <circle fill="#FFFFFF" cx="41.353"  cy="81.370" r="2.986"/>
  <circle fill="#FFFFFF" cx="28.317"  cy="79.029" r="2.986"/>
  <circle fill="#FFFFFF" cx="18.165"  cy="68.870" r="2.986"/>
  <circle fill="#FFFFFF" cx="9.872"   cy="53.211" r="2.985"/>
  <circle fill="#FFFFFF" cx="91.695"  cy="30.742" r="2.986"/>
  <circle fill="#FFFFFF" cx="97.667"  cy="33.655" r="2.986"/>
  <circle fill="#FFFFFF" cx="107.515" cy="46.101" r="2.986"/>
  <circle fill="#FFFFFF" cx="119.803" cy="69.671" r="2.985"/>
  <circle fill="#FFFFFF" cx="146.330" cy="80.016" r="2.986"/>
  <circle fill="#FFFFFF" cx="162.145" cy="59.119" r="2.985"/>
  <circle fill="#FFFFFF" cx="158.145" cy="66.685" r="2.985"/>
  <circle fill="#FFFFFF" cx="175.993" cy="36.769" r="2.985"/>
  <circle fill="#FFFFFF" cx="198.840" cy="32.678" r="2.986"/>
  <circle fill="#FFFFFF" cx="208.491" cy="45.121" r="2.985"/>
  <g class="lc-feat-s1">
    <line stroke="#94CEC8" stroke-width="1.049" stroke-dasharray="3.147,3.147" x1="9.873" y1="53.211" x2="210.191" y2="53.211"/>
    <line stroke="#F3BEA8" stroke-width="1.119" x1="222.316" y1="31.429" x2="222.316" y2="53.212"/>
    <line stroke="#F3BEA8" stroke-width="1.119" x1="219.379" y1="31.429" x2="225.255" y2="31.429"/>
    <line stroke="#F3BEA8" stroke-width="1.119" x1="219.379" y1="53.212" x2="225.255" y2="53.212"/>
    <text x="229.221" y="45.121" fill="#F3BEA8" font-family="'Inter',sans-serif" font-size="11.193">A</text>
  </g>
  <g class="lc-feat-s2">
    <line stroke="#ACDEC7" stroke-width="1.119" x1="91.694" y1="21.689" x2="192.867" y2="21.689"/>
    <line stroke="#ACDEC7" stroke-width="1.119" x1="91.694" y1="18.751" x2="91.694"  y2="24.627"/>
    <line stroke="#ACDEC7" stroke-width="1.119" x1="192.867" y1="18.751" x2="192.867" y2="24.627"/>
    <text x="138.545" y="17.371" fill="#ACDEC7" font-family="'Inter',sans-serif" font-size="11.193">P</text>
  </g>
</svg>
</div>
<div class="lc-card-body">
<h3><a href="features/">Feature extractors</a></h3>
<p>40+ features: magnitude and flux statistics, time-series shape descriptors, period extraction, and parametric fits for transients. Supports multi-band light curves and optimized to process 10⁶–10⁹ objects.</p>
</div>
</div>

<div class="lc-card">
<div class="lc-card-svg-wrap">
<svg class="lc-svg-embed" viewBox="0 0 240 110" fill="none" xmlns="http://www.w3.org/2000/svg">
<g transform="translate(9,0)">
  <path stroke="#FFFFFF" d="M2.869,60.45c3.136-0.437,8.568-26.801,16.048-26.801c7.259,0,30.56,26.438,44.583,26.512c0,0,1.058,0.089,2.103-0.5"/>
  <circle fill="#FFFFFF" cx="65.136" cy="60.162" r="1.581"/>
  <circle fill="#FFFFFF" cx="58.332" cy="58.872" r="1.578"/>
  <circle fill="#FFFFFF" cx="41.967" cy="49.375" r="1.579"/>
  <circle fill="#FFFFFF" cx="32.215" cy="41.937" r="1.579"/>
  <circle fill="#FFFFFF" cx="20.939" cy="34.252" r="1.579"/>
  <circle fill="#FFFFFF" cx="14.892" cy="36.131" r="1.579"/>
  <circle fill="#FFFFFF" cx="8.282"  cy="48.596" r="1.579"/>
  <circle fill="#FFFFFF" cx="2.869"  cy="60.162" r="1.580"/>
  <circle fill="#FFFFFF" cx="26.909" cy="37.979" r="1.579"/>
  <line stroke="#94CEC8" x1="69.722" y1="55" x2="82.187" y2="55"/>
  <polygon fill="#94CEC8" points="79.537,58.271 78.991,57.686 81.880,55.001 78.991,52.315 79.537,51.729 83.055,55.001"/>
  <circle fill="#FFFFFF" opacity="0.5" cx="95.911" cy="27.875" r="5.5"/>
  <circle fill="#FFFFFF" opacity="0.7" cx="95.911" cy="55"     r="5.5"/>
  <circle fill="#FFFFFF" opacity="0.5" cx="95.911" cy="82.125" r="5.5"/>
  <circle fill="#FFFFFF" opacity="0.5" cx="140.327" cy="19.880" r="4.634"/>
  <circle fill="#FFFFFF" opacity="0.7" cx="140.327" cy="43.293" r="4.634"/>
  <circle fill="#FFFFFF" opacity="0.7" cx="140.327" cy="66.707" r="4.634"/>
  <circle fill="#FFFFFF" opacity="0.5" cx="140.327" cy="90.120" r="4.634"/>
  <circle fill="#FFFFFF" cx="177.787" cy="55" r="6.655"/>
  <g class="lc-conn-a">
    <polyline stroke="#FFFFFF" stroke-linecap="round" stroke-linejoin="round" points="135.527,20.213 101.328,27.875 135.452,43.293 101.494,55 135.370,66.707 101.577,82.125 135.548,89.922"/>
    <polyline stroke="#FFFFFF" stroke-linecap="round" stroke-linejoin="round" points="135.527,20.213 101.328,55 135.548,89.922"/>
    <line  stroke="#FFFFFF" stroke-linecap="round" stroke-linejoin="round" x1="135.548" y1="89.922" x2="101.328" y2="27.875"/>
    <polyline stroke="#FFFFFF" stroke-linejoin="round" points="135.527,20.213 101.411,82.125 135.693,43.293"/>
    <line  stroke="#FFFFFF" stroke-linejoin="round" x1="135.370" y1="66.707" x2="101.328" y2="27.875"/>
  </g>
  <g class="lc-conn-b">
    <polyline stroke="#FFFFFF" stroke-linecap="round" stroke-linejoin="round" points="144.961,19.880 171.132,55.278 144.961,90.120"/>
    <polyline stroke="#FFFFFF" stroke-linecap="round" stroke-linejoin="round" points="144.961,43.293 171.132,55.278 144.961,66.707"/>
  </g>
  <g class="lc-vec-text">
    <text x="189" y="21.667" fill="#F3BEA8" font-family="'JetBrains Mono',monospace" font-size="8">&nbsp;0.73,</text>
    <text x="189" y="33.667" fill="#F3BEA8" font-family="'JetBrains Mono',monospace" font-size="8">&nbsp;0.31,</text>
    <text x="189" y="45.667" fill="#F3BEA8" font-family="'JetBrains Mono',monospace" font-size="8">&#x2013;1.22,</text>
    <text x="189" y="57.667" fill="#F3BEA8" font-family="'JetBrains Mono',monospace" font-size="8">&nbsp;0.84,</text>
    <text x="189" y="69.667" fill="#F3BEA8" font-family="'JetBrains Mono',monospace" font-size="8">&#x2013;0.28,</text>
    <text x="189" y="81.667" fill="#F3BEA8" font-family="'JetBrains Mono',monospace" font-size="8">&nbsp;...,</text>
    <text x="189" y="93.667" fill="#F3BEA8" font-family="'JetBrains Mono',monospace" font-size="8">&nbsp;2.03</text>
  </g>
</g>
</svg>
</div>
<div class="lc-card-body">
<h3><a href="embed/">ML embeddings</a></h3>
<p>Map raw light curves to dense vectors using pretrained transformer models. Suitable for classification, anomaly detection, and similarity search at different scales</p>
</div>
</div>

<div class="lc-card">
<div class="lc-card-svg-wrap">
<svg class="lc-svg-dmdt" viewBox="0 0 240 110" fill="none" xmlns="http://www.w3.org/2000/svg">
  <polyline stroke="#FFFFFF" stroke-linejoin="round" points="14.684,45.373 30.101,70.035 45.267,52.785 60.601,35.535 75.851,59.618"/>
  <circle fill="#FFFFFF" cx="14.684" cy="45.373" r="3.184"/>
  <circle fill="#FFFFFF" cx="30.101" cy="70.035" r="3.184"/>
  <circle fill="#FFFFFF" cx="45.267" cy="52.785" r="3.184"/>
  <circle fill="#FFFFFF" cx="60.601" cy="35.535" r="3.184"/>
  <circle fill="#FFFFFF" cx="75.851" cy="59.618" r="3.184"/>
  <line stroke="#94CEC8" x1="82.768" y1="53.25" x2="103.899" y2="53.25"/>
  <polygon fill="#94CEC8" points="101.249,56.522 100.703,55.936 103.592,53.251 100.703,50.565 101.249,49.979 104.767,53.251"/>
  <path opacity="0.2" fill="#FFFFFF" d="M114.225,12.315c0-.55.451-1,1-1h13.834c.549,0,1,.45,1,1v13.833c0,.55-.451,1-1,1h-13.834c-.549,0-1-.45-1-1z"/>
  <path opacity="0.4" fill="#FFFFFF" d="M132.381,12.315c0-.55.451-1,1-1h13.834c.549,0,1,.45,1,1v13.833c0,.55-.451,1-1,1h-13.834c-.549,0-1-.45-1-1z"/>
  <path opacity="0.3" fill="#FFFFFF" d="M150.538,12.315c0-.55.451-1,1-1h13.834c.549,0,1,.45,1,1v13.833c0,.55-.451,1-1,1h-13.834c-.549,0-1-.45-1-1z"/>
  <path opacity="0.2" fill="#FFFFFF" d="M168.694,12.315c0-.55.451-1,1-1h13.834c.549,0,1,.45,1,1v13.833c0,.55-.451,1-1,1h-13.834c-.549,0-1-.45-1-1z"/>
  <path opacity="0.1" fill="#FFFFFF" d="M186.85,12.315c0-.55.451-1,1-1h13.834c.549,0,1,.45,1,1v13.833c0,.55-.451,1-1,1H187.85c-.549,0-1-.45-1-1z"/>
  <path opacity="0.3" fill="#FFFFFF" d="M114.225,30.87c0-.55.451-1,1-1h13.834c.549,0,1,.45,1,1v13.833c0,.549-.451,1-1,1h-13.834c-.549,0-1-.451-1-1z"/>
  <path opacity="0.5" fill="#FFFFFF" d="M132.381,30.87c0-.55.451-1,1-1h13.834c.549,0,1,.45,1,1v13.833c0,.549-.451,1-1,1h-13.834c-.549,0-1-.451-1-1z"/>
  <path opacity="0.7" fill="#FFFFFF" d="M150.538,30.87c0-.55.451-1,1-1h13.834c.549,0,1,.45,1,1v13.833c0,.549-.451,1-1,1h-13.834c-.549,0-1-.451-1-1z"/>
  <path opacity="0.5" fill="#FFFFFF" d="M168.694,30.87c0-.55.451-1,1-1h13.834c.549,0,1,.45,1,1v13.833c0,.549-.451,1-1,1h-13.834c-.549,0-1-.451-1-1z"/>
  <path opacity="0.3" fill="#FFFFFF" d="M186.85,30.87c0-.55.451-1,1-1h13.834c.549,0,1,.45,1,1v13.833c0,.549-.451,1-1,1H187.85c-.549,0-1-.451-1-1z"/>
  <path opacity="0.2" fill="#FFFFFF" d="M114.225,49.424c0-.551.451-1,1-1h13.834c.549,0,1,.449,1,1v13.832c0,.551-.451,1-1,1h-13.834c-.549,0-1-.449-1-1z"/>
  <path opacity="0.5" fill="#FFFFFF" d="M132.381,49.424c0-.551.451-1,1-1h13.834c.549,0,1,.449,1,1v13.832c0,.551-.451,1-1,1h-13.834c-.549,0-1-.449-1-1z"/>
  <path opacity="0.6" fill="#FFFFFF" d="M150.538,49.424c0-.551.451-1,1-1h13.834c.549,0,1,.449,1,1v13.832c0,.551-.451,1-1,1h-13.834c-.549,0-1-.449-1-1z"/>
  <path opacity="0.6" fill="#FFFFFF" d="M168.694,49.424c0-.551.451-1,1-1h13.834c.549,0,1,.449,1,1v13.832c0,.551-.451,1-1,1h-13.834c-.549,0-1-.449-1-1z"/>
  <path opacity="0.4" fill="#FFFFFF" d="M186.85,49.424c0-.551.451-1,1-1h13.834c.549,0,1,.449,1,1v13.832c0,.551-.451,1-1,1H187.85c-.549,0-1-.449-1-1z"/>
  <path opacity="0.1" fill="#FFFFFF" d="M114.225,67.978c0-.55.451-1,1-1h13.834c.549,0,1,.45,1,1v13.833c0,.55-.451,1-1,1h-13.834c-.549,0-1-.45-1-1z"/>
  <path opacity="0.2" fill="#FFFFFF" d="M132.381,67.978c0-.55.451-1,1-1h13.834c.549,0,1,.45,1,1v13.833c0,.55-.451,1-1,1h-13.834c-.549,0-1-.45-1-1z"/>
  <path opacity="0.3" fill="#FFFFFF" d="M150.538,67.978c0-.55.451-1,1-1h13.834c.549,0,1,.45,1,1v13.833c0,.55-.451,1-1,1h-13.834c-.549,0-1-.45-1-1z"/>
  <path opacity="0.5" fill="#FFFFFF" d="M168.694,67.978c0-.55.451-1,1-1h13.834c.549,0,1,.45,1,1v13.833c0,.55-.451,1-1,1h-13.834c-.549,0-1-.45-1-1z"/>
  <path opacity="0.5" fill="#FFFFFF" d="M186.85,67.978c0-.55.451-1,1-1h13.834c.549,0,1,.45,1,1v13.833c0,.55-.451,1-1,1H187.85c-.549,0-1-.45-1-1z"/>
  <text x="210.538" y="49.731" fill="#94CEC8" font-family="'Inter',sans-serif" font-size="10">Δm</text>
  <text x="147.632" y="95.773" fill="#94CEC8" font-family="'Inter',sans-serif" font-size="10">lg Δt</text>
  <g class="lc-dmdt-s1">
    <line stroke="#FF8000" x1="14.684" y1="45.373" x2="29.797" y2="70.036"/>
    <circle fill="#FF8000" cx="14.684" cy="45.373" r="3.184"/>
    <circle fill="#FF8000" cx="30.101" cy="70.161" r="3.184"/>
    <path fill="#FF8000" d="M114.225,67.978c0-.55.451-1,1-1h13.834c.549,0,1,.45,1,1v13.833c0,.55-.451,1-1,1h-13.834c-.549,0-1-.45-1-1z"/>
  </g>
  <g class="lc-dmdt-s2">
    <line stroke="#2CB0CF" x1="14.684" y1="45.373" x2="60.601" y2="35.535"/>
    <circle fill="#2CB0CF" cx="14.684" cy="45.373" r="3.184"/>
    <circle fill="#2CB0CF" cx="60.601" cy="35.535" r="3.184"/>
    <path fill="#2CB0CF" d="M150.538,30.87c0-.55.451-1,1-1h13.834c.549,0,1,.45,1,1v13.833c0,.549-.451,1-1,1h-13.834c-.549,0-1-.451-1-1z"/>
  </g>
  <g class="lc-dmdt-s3">
    <line stroke="#AB3D9B" x1="45.267" y1="52.785" x2="75.851" y2="59.772"/>
    <circle fill="#AB3D9B" cx="45.267" cy="52.785" r="3.184"/>
    <circle fill="#AB3D9B" cx="75.851" cy="59.618" r="3.184"/>
    <path fill="#AB3D9B" d="M114.225,49.424c0-.551.451-1,1-1h13.834c.549,0,1,.449,1,1v13.832c0,.551-.451,1-1,1h-13.834c-.549,0-1-.449-1-1z"/>
  </g>
</svg>
</div>
<div class="lc-card-body">
<h3><a href="dmdt/">dm-dt maps</a></h3>
<p>2D histograms of Δmag vs log-Δt for all observation pairs, providing a fixed-size image representation for CNN-based variability classifiers.</p>
</div>
</div>

</div>

## Quick start

```python
import light_curve as lc
from light_curve.embed import Astromer2
import numpy as np

rng = np.random.default_rng(0)
t = np.sort(rng.uniform(0, 100, 100))
m = 15.0 + 0.01 * t + rng.normal(0, 0.1, 100)
err = np.full(100, 0.1)

# Feature extraction
extractor = lc.Extractor(lc.Amplitude(), lc.BeyondNStd(nstd=1), lc.LinearFit())
result = extractor(t, m, err)

# ML embedding with pretrained Astromer2 (downloads on first use)
model = Astromer2.from_hf(output="mean")
embedding = model(t, m).squeeze()   # shape (256,)

# dm-dt map for CNN-based variability classifiers
dmdt = lc.DmDt.from_borders(min_lgdt=0, max_lgdt=2, max_abs_dm=1.0, lgdt_size=16, dm_size=16, norm=[])
matrix = dmdt.points(t, m)   # shape (16, 16)
```

---

## Used by

<div class="lc-used-by">

<div class="lc-ub-col">
<section class="lc-ub-section">
<h3 class="lc-ub-heading">Alert brokers</h3>
<p class="lc-ub-desc">AMPEL, ANTARES, and Fink use <code>light-curve</code> for real-time feature extraction
when classifying on the order of a million alerts per night from the Zwicky Transient Facility
and the Rubin Observatory.</p>
<div class="lc-ub-logos">
  <a href="https://ampelproject.github.io" target="_blank" rel="noopener" class="lc-broker-logo">
    <img src="assets/brokers/ampel.png" alt="AMPEL">
  </a>
  <a href="https://antares.noirlab.edu" target="_blank" rel="noopener" class="lc-broker-logo">
    <img src="assets/brokers/antares.svg" alt="ANTARES">
  </a>
  <a href="https://fink-broker.org" target="_blank" rel="noopener" class="lc-broker-logo">
    <img src="assets/brokers/fink.svg" alt="Fink">
  </a>
</div>
</section>

<section class="lc-ub-section">
<h3 class="lc-ub-heading">The SNAD team</h3>
<a href="https://snad.space" target="_blank" rel="noopener" class="lc-broker-logo">
  <img src="assets/brokers/snad.png" alt="SNAD">
</a>
<p class="lc-ub-desc">The <a href="https://snad.space" target="_blank" rel="noopener">SNAD</a>
anomaly-detection group uses <code>light-curve</code> to analyze hundreds of millions of
Zwicky Transient Facility light curves from data releases, powering large-scale analyses
and the public <a href="https://ztf.snad.space" target="_blank" rel="noopener">SNAD ZTF Viewer</a>.</p>
</section>
</div>

<div class="lc-ub-col">
<section class="lc-ub-section">
<h3 class="lc-ub-heading">Software</h3>
<div class="lc-ub-pkg-grid">
  <div class="lc-ub-pkg">
    <div class="lc-ub-pkg-logo-wrap"><a href="https://github.com/quatrope/feets" target="_blank" rel="noopener"><img src="assets/packages/feets.svg" alt="feets" class="lc-ub-pkg-logo"></a></div>
    <span class="lc-ub-pkg-desc">Feature extraction library for time series with Dask-parallel processing and a scikit-learn-style API, using light-curve as its computational backend</span>
  </div>
  <div class="lc-ub-pkg">
    <div class="lc-ub-pkg-logo-wrap"><a href="https://github.com/COINtoolbox/RESSPECT" target="_blank" rel="noopener"><img src="assets/packages/resspect.png" alt="RESSPECT" class="lc-ub-pkg-logo"></a></div>
    <span class="lc-ub-pkg-desc">Photometric supernova classification pipeline for LSST, with active learning, developed by LSST DESC and COIN</span>
  </div>
  <div class="lc-ub-pkg">
    <div class="lc-ub-pkg-logo-wrap"><a href="https://github.com/VTDA-Group/superphot-plus" target="_blank" rel="noopener"><img src="assets/packages/superphot-plus.svg" alt="superphot+" class="lc-ub-pkg-logo lc-ub-pkg-logo--wordmark"></a></div>
    <span class="lc-ub-pkg-desc">Real-time supernova light curve fitting and classification for ZTF/Rubin</span>
  </div>
</div>
</section>
</div>

</div>

<section class="lc-ub-section">
<h3 class="lc-ub-heading">Publications</h3>
<p class="lc-ub-desc"><strong>Used in <a href="https://ui.adsabs.harvard.edu/public-libraries/hcwBtIKwQ3yJjT784upA7A" target="_blank" rel="noopener">30+ research publications</a>.</strong></p>
</section>
