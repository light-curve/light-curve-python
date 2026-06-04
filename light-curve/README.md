# `light-curve` processing toolbox for Python

A Python wrapper for the [`light-curve-feature`](https://github.com/light-curve/light-curve-feature) and
[`light-curve-dmdt`](https://github.com/light-curve/light-curve-dmdt) Rust crates, providing high-performance
time-series feature extraction for astrophysics.

[![PyPI version](https://badge.fury.io/py/light-curve.svg)](https://pypi.org/project/light-curve/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/light-curve-python.svg)](https://anaconda.org/conda-forge/light-curve-python)
[![testing](https://github.com/light-curve/light-curve-python/actions/workflows/test.yml/badge.svg)](https://github.com/light-curve/light-curve-python/actions/workflows/test.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/light-curve/light-curve-python/main.svg)](https://results.pre-commit.ci/latest/github/light-curve/light-curve-python/main)

**Full documentation:** [light-curve.snad.space](https://light-curve.snad.space/)

## Quick start

```sh
pip install 'light-curve[full]'
```

<!-- name: test_quickstart -->

```python
import light_curve as licu
import numpy as np

rng = np.random.default_rng(0)
t = np.sort(rng.uniform(0, 100, 100))
m = 15.0 + 0.01 * t + rng.normal(0, 0.1, 100)
err = np.full(100, 0.1)

# Statistical and variability feature extraction
extractor = licu.Extractor(licu.Amplitude(), licu.BeyondNStd(nstd=1), licu.LinearFit())
result = extractor(t, m, err)
print('\n'.join(f"{name} = {value:.4f}" for name, value in zip(extractor.names, result)))

# Multi-band: per-band features and cross-band color in one Extractor
band = np.tile(["g", "r"], 50)  # 50 interleaved g/r observations
ext_mb = licu.Extractor(licu.WeightedMean(bands=["g", "r"]), licu.ColorOfMedian(["g", "r"]))
result_mb = ext_mb(t, m, err, band=band)
print(dict(zip(ext_mb.names, result_mb)))

# dm-dt map — 2D histogram of Δmag vs log-Δt for CNN classifiers
dmdt = licu.DmDt.from_borders(min_lgdt=0, max_lgdt=2, max_abs_dm=1.0, lgdt_size=16, dm_size=16, norm=[])
matrix = dmdt.points(t, m)
print(f"dm-dt map shape: {matrix.shape}")  # (16, 16)
```

Embed the light curve with a pretrained Astromer2 transformer (model downloads on first use):

```python
from light_curve.embed import Astromer2
import numpy as np

rng = np.random.default_rng(0)
t = np.sort(rng.uniform(0, 100, 100))
m = 15.0 + 0.01 * t + rng.normal(0, 0.1, 100)

model = Astromer2.from_hf(output="mean")  # cached after first download
embedding = model(t, m).squeeze()  # shape (256,)
print(f"Embedding shape: {embedding.shape}")
```

The package provides:

- **40+ feature extractors** — amplitude, period, variability indices, parametric fits, and more.
  See the [Features docs](https://light-curve.snad.space/features/) for a full list.
- **ML embeddings** — embed light curves with pretrained ONNX models (Astromer2, ATCAT).
  See the [Embeddings docs](https://light-curve.snad.space/embed/).
- **dm-dt maps** — 2D histograms of Δm vs log-Δt for CNN input.
  See the [dm-dt docs](https://light-curve.snad.space/dmdt/).

## Installation

```sh
pip install 'light-curve[full]'
# or
conda install -c conda-forge light-curve-python
```

The `full` extra installs optional Python dependencies required by some features
(`iminuit`, `scipy`).

Binary wheels are available for Linux (x86-64, aarch64), macOS, and Windows.
For the full platform support table and build-from-source instructions, see the
[Installation docs](https://light-curve.snad.space/installation/).

## Developer guide

### Prepare environment

Install a recent Rust toolchain and Python 3.10+.
Use [`rustup`](https://rustup.rs/) to install Rust and keep it updated with `rustup update`.

Clone the repository, then create and activate a virtual environment:

```bash
git clone https://github.com/light-curve/light-curve-python.git
cd light-curve-python/light-curve
python3 -m venv venv
source venv/bin/activate
```

Install the package in editable mode:

```bash
python -mpip install maturin
maturin develop --group dev
```

Run this command on initial setup. On subsequent runs, activate the environment with
`source venv/bin/activate` and rebuild Rust code with `maturin develop`.
Python-only changes require no rebuild.

Install pre-commit hooks:

```bash
pre-commit install
```

### Run tests

```bash
python -mpytest
```

Benchmarks are disabled by default; enable them with `--benchmark-enable`.

## Citation

If you found this project useful for your research, please
cite [Malanchev et al., 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.502.5147M/abstract):

```bibtex
@ARTICLE{2021MNRAS.502.5147M,
       author = {{Malanchev}, K.~L. and {Pruzhinskaya}, M.~V. and {Korolev}, V.~S. and {Aleo}, P.~D. and {Kornilov}, M.~V. and {Ishida}, E.~E.~O. and {Krushinsky}, V.~V. and {Mondon}, F. and {Sreejith}, S. and {Volnova}, A.~A. and {Belinski}, A.~A. and {Dodin}, A.~V. and {Tatarnikov}, A.~M. and {Zheltoukhov}, S.~G. and {(The SNAD Team)}},
        title = "{Anomaly detection in the Zwicky Transient Facility DR3}",
      journal = {\mnras},
     keywords = {methods: data analysis, astronomical data bases: miscellaneous, stars: variables: general, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Solar and Stellar Astrophysics},
         year = 2021,
        month = apr,
       volume = {502},
       number = {4},
        pages = {5147-5175},
          doi = {10.1093/mnras/stab316},
archivePrefix = {arXiv},
       eprint = {2012.01419},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021MNRAS.502.5147M},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
