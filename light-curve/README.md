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

<!-- name: test_feature_evaluators_basic -->

```python
import light_curve as lc
import numpy as np

rng = np.random.default_rng(0)
n = 100  # observations per light curve

# Observation times in days (unevenly sampled, must be sorted)
t = np.sort(rng.uniform(0, 100, n))
# Magnitudes with a slight linear fade and measurement noise
m = 15.0 + 0.01 * t + rng.normal(0, 0.1, n)
err = np.full(n, 0.1)

# Combine features into a single extractor evaluated in one pass
extractor = lc.Extractor(lc.Amplitude(), lc.BeyondNStd(nstd=1), lc.LinearFit())

result = extractor(t, m, err)
print('\n'.join(f"{name} = {value:.4f}" for name, value in zip(extractor.names, result)))

# Extract a feature from 1000 light curves in parallel
light_curves = [
    (np.sort(rng.uniform(0, 100, n)), 15.0 + rng.normal(0, 0.2, n), np.full(n, 0.1))
    for _ in range(1000)
]
amplitudes = lc.Amplitude().many(light_curves)
print(f"Amplitude: mean = {np.mean(amplitudes):.3f} mag, std = {np.std(amplitudes):.3f} mag")
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

Binary wheels are available on [PyPI](https://pypi.org/project/light-curve/) and
[Anaconda](https://anaconda.org/conda-forge/light-curve-python) for common platforms:

| Arch \ OS   | Linux glibc 2.17+ | Linux musl 1.2+                | macOS                 | Windows                     |
|-------------|-------------------|--------------------------------|-----------------------|-----------------------------|
| **x86-64**  | PyPI (MKL), conda | PyPI (MKL)                     | PyPI macOS 15+, conda | PyPI, conda (conda: no GSL) |
| **i686**    | src               | src                            | —                     | not tested                  |
| **aarch64** | PyPI              | PyPI                           | PyPI macOS 14+, conda | not tested                  |
| **ppc64le** | src               | not tested (no Rust toolchain) | —                     | —                           |

- **PyPI / conda**: A binary wheel or package is available on pypi.org or anaconda.org.
  Local building is not required; the only prerequisite is a recent version of `pip` or `conda`.
  For Linux x86-64, PyPI wheels are built with Intel MKL for improved periodogram performance,
  which is not the default build option.
  For Windows x86-64, the conda package excludes GSL support (LMSDER fitting unavailable);
  PyPI wheels include both Ceres and GSL.
- **src**: The package has been confirmed to build and pass unit tests locally,
  but CI does not test or publish packages for this platform.
  See the [Build from source](#build-from-source) section for details.
  Please open an issue or pull request if you encounter build problems or would like us to distribute
  packages for these platforms.
- **not tested**: Building from source has not been tested.
  Please report build status via issue, PR, or email.

macOS wheels require relatively recent OS versions; please open an issue if you need support for older macOS
versions (see https://github.com/light-curve/light-curve-python/issues/376 for details).

We no longer publish PyPy wheels ([#345](https://github.com/light-curve/light-curve-python/issues/345))
or the PPC64le CPython glibc wheel ([#479](https://github.com/light-curve/light-curve-python/issues/479));
please open an issue if you need either.

Free-threaded Python is supported when built from source for Python 3.14+
([officially supported in Python 3.14+](https://docs.python.org/3.14/whatsnew/3.14.html#whatsnew314-pep779)).
No pre-built distributions are currently provided; please comment on the relevant issues if you need them:
[PyPI binary wheel issue](https://github.com/light-curve/light-curve-python/issues/500),
[conda-forge package issue](https://github.com/conda-forge/light-curve-python-feedstock/issues/11).
Note that for expensive features, the GIL-enabled interpreter with `.many()` achieves performance on par with
the free-threaded interpreter using Python threads. For inexpensive extractors, `.many()` still reduces
Rust–Python interaction overhead significantly.

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

### Build from source

The package supports several Cargo features controlling optional C/C++ dependencies:

| Feature | Default | Description |
|---------|---------|-------------|
| `abi3` | yes | Stable CPython ABI3 — disable for PyPy or free-threading |
| `ceres-source` | yes | Build Ceres optimizer from source (requires C++ and CMake) |
| `ceres-system` | no | Link against system Ceres instead |
| `mkl` | no | Intel MKL backend for the fast periodogram (recommended for Intel CPUs) |
| `gsl` | yes | GNU Scientific Library support (requires GSL ≥ 2.5) |
| `mimalloc` | yes | mimalloc allocator (up to 2× faster for cheap features) |

Minimal build (no system C/C++ dependencies):

```bash
python -mpip install maturin
maturin build --release --locked --no-default-features --features=abi3,mimalloc
```

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
