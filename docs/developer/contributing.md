---
hide:
  - navigation
---

# Contributing

## Environment setup

Requires:

- **Rust** ≥ 1.85 — install via [rustup](https://rustup.rs/) and keep updated with `rustup update`
- **Python** ≥ 3.10
- **CMake** — needed to build the Ceres solver (skip with `--no-default-features` if you don't need fitting features)
- **GSL** (optional) — enables the `"lmsder"` fitting backend; install via your system package manager (`libgsl-dev` on Debian/Ubuntu, `gsl` on Homebrew)

```bash
git clone https://github.com/light-curve/light-curve-python.git
cd light-curve-python/light-curve

python3 -m venv venv && source venv/bin/activate
pip install maturin
maturin develop --group dev   # builds Rust extension + installs dev deps
pre-commit install            # install pre-commit hooks
```

After this initial setup: activate the venv with `source venv/bin/activate`, then rebuild
Rust code with `maturin develop` after any Rust change. Python-only changes need no rebuild.

For a faster iteration cycle (no system deps):

```bash
maturin develop --no-default-features --features=abi3,mimalloc
```

## Running tests

```bash
pytest                                               # all tests (benchmarks disabled by default)
pytest --benchmark-enable                            # with benchmarks
pytest tests/light_curve_ext/test_feature.py        # single test file
pytest -k "Amplitude"                               # filter by name
```

Tests also cover the README examples via `markdown-pytest`.

## Linting and formatting

All linting runs automatically via pre-commit hooks on each commit.

```bash
# Python
ruff check .            # lint
ruff format .           # format

# Rust
cargo fmt --manifest-path=Cargo.toml
cargo clippy --all-targets -- -D warnings   # warnings are errors

# Everything at once
pre-commit run --all-files
```

## Cargo features

| Feature | Default | Effect |
|---------|---------|--------|
| `abi3` | ✓ | Stable CPython ABI3 — disable for PyPy or free-threaded builds |
| `ceres-source` | ✓ | Build Ceres solver from source (requires C++ compiler + CMake) |
| `ceres-system` | | Link against system-installed Ceres instead of building from source |
| `gsl` | ✓ | Enable GSL backend for `BazinFit`/`VillarFit` (requires `libgsl-dev`) |
| `mkl` | | Intel MKL FFTW backend for the fast periodogram (strongly recommended for Intel CPUs) |
| `mimalloc` | ✓ | mimalloc memory allocator — up to 2× speedup for cheap features |

## Architecture

The package has a layered import structure:

1. `light_curve.light_curve_py` — pure-Python experimental implementations
2. `light_curve.light_curve_ext` — compiled Rust extension (via PyO3/Maturin)
3. `light_curve.__init__` — imports Python implementations first, then overrides with faster Rust equivalents

This means every extractor has a Python fallback, but the Rust version takes precedence at runtime.

**Rust source layout** (inside `light-curve/src/`):

| File | Contents |
|------|---------|
| `features.rs` | All 40+ feature extractors, PyO3 class definitions |
| `dmdt.rs` | dm-dt map implementation |
| `ln_prior.rs` | Log-prior helpers for MCMC fitting |
| `lib.rs` | Module exports and top-level PyO3 wiring |

## Adding a new feature extractor

1. Implement the feature in the upstream [`light-curve-feature`](https://github.com/light-curve/light-curve-feature) Rust crate.
2. Add PyO3 bindings in `src/features.rs` and export in `src/lib.rs`.
3. Optionally add a pure-Python experimental version in `light_curve/light_curve_py/features/`.
4. Add tests in `tests/light_curve_ext/test_feature.py`.
5. Add benchmarks in `tests/test_w_bench.py`.
6. Update `CHANGELOG.md`.

## Free-threaded Python

Free-threaded CPython (PEP 703, available from Python 3.13t / 3.14t) is supported when
built from source. The standard `maturin develop` builds work, but use
`--group dev-free-threading` instead of `--group dev` to avoid packages that don't yet
support free threading (iminuit, polars, cesium).

Pre-built free-threaded wheels are not yet published — track progress at
[#500](https://github.com/light-curve/light-curve-python/issues/500).

## Publishing a release

1. Update `CHANGELOG.md` with the new version and date.
2. Update the version in `Cargo.toml` (the Python package version is read from there).
3. Create and push a git tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.
4. CI (`publish.yml`) will build wheels via cibuildwheel and upload to PyPI.
