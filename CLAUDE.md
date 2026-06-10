# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`light-curve-python` is a hybrid Rust/Python package for time-series feature extraction in astrophysics. It wraps the
`light-curve-feature` and `light-curve-dmdt` Rust crates via PyO3/Maturin bindings.

**Default branch is `main`** (not `master`). Use `main` as the base for PRs and release branches.

## Repository Layout

The main package lives in `light-curve/` (not the repo root). There's also a thin alias package in
`light-curve-python/`.

Within `light-curve/`:

- `src/` — Rust PyO3 bindings (`features.rs` is the main file with 40+ feature extractors, `dmdt.rs` for dm-dt maps)
- `light_curve/` — Python package
    - `light_curve_py/` — Pure Python experimental feature implementations
    - `light_curve_ext.py` — Re-exports from compiled Rust extension
    - `__init__.py` — Imports Python features first, then overrides with faster Rust equivalents
- `tests/` — pytest suite

## Build & Development Commands

All commands run from `light-curve/` directory:

```bash
# Setup
pip install maturin
maturin develop --group=dev           # Dev build with test dependencies
maturin develop --release             # Optimized build

# After Rust changes, re-run maturin develop. Python-only changes need no rebuild.

# Minimal build (fewer system deps)
maturin build --release --locked --no-default-features --features=abi3,mimalloc
```

## Testing

```bash
pytest                                                    # All tests (benchmarks disabled by default)
pytest --benchmark-enable                                 # With benchmarks
pytest -m "not (nobs or multi)" --benchmark-enable        # Skip slow benchmarks
pytest tests/light_curve_ext/test_feature.py              # Single test file
pytest tests/light_curve_ext/test_feature.py::test_name   # Single test
```

Tests also run against README.md examples via `markdown-pytest`. Test paths: `tests/`, `README.md`.

## Linting & Formatting

```bash
# Python
ruff check .              # Lint
ruff format .             # Format

# Rust
cargo fmt --manifest-path=Cargo.toml
cargo clippy --all-targets -- -D warnings     # Warnings are errors

# All at once
pre-commit run --all-files
```

## Architecture Notes

**Import layering**: `__init__.py` imports all Python implementations first, then overwrites them with Rust equivalents.
This means Rust features shadow Python ones at runtime, but both exist for testing/comparison.

**Import alias**: All docs and examples use `import light_curve as licu` (not `lc`).

**Feature extraction pattern**: Each feature (e.g., `Amplitude`, `BazinFit`, `Periodogram`) is a class with
`__call__(t, m, sigma, ...)` for single light curves and `.many(...)` for batch processing with reduced Python-Rust
overhead.

**Multiband mode**: Every Rust-backed feature accepts a `bands=["g", "r", ...]` constructor argument. When set,
`__call__` expects a fourth `band` string array; output names get a passband suffix (e.g. `amplitude_g`).
Pure-multiband features (`ColorOfMaximum`, `ColorOfMedian`, `ColorOfMinimum`, `ColorSpread`) take `bands` as their
first positional argument and have no single-band mode. `Periodogram` supports `bands=` via `MultiColorPeriodogram`
(add `multiband_normalization=`). `Extractor` freely mixes single-band and multiband features; it filters the band
array automatically for each sub-feature.

**Rust edition**: 2024, MSRV 1.85. Clippy treats warnings as errors.

**Cargo features**: `abi3` (stable Python ABI), `ceres-source`/`ceres-system`, `mkl` (Intel MKL for FFTW-based fast
periodogram), `gsl`, `mimalloc`. Default features include abi3, ceres-source, gsl, mimalloc.

**Code style**: Line length 120, Python target 3.10+, Ruff for Python linting (rules: F, E, W, I001, NPY201).

**Rust style**: Write idiomatic Rust — prefer immutability by default (`let` over `let mut`), use iterators and
combinators (`.map()`, `.filter()`, `.collect()`, `.fold()`, etc.) instead of imperative `for` loops with manual
accumulation, leverage pattern matching and `if let`/`let else` over nested `if`/`else` chains, use `?` for error
propagation, and prefer owned types only when necessary (borrow when possible).

## Adding a New Feature Extractor

1. Core implementation goes in upstream `light-curve-feature` Rust crate
2. Add PyO3 bindings in `src/features.rs`, export in `src/lib.rs`
   - For pure-multiband features, use the `color_two_band_feature!` macro (2-band) or model after `ColorSpread` (≥2 bands)
3. Optionally add experimental Python version in `light_curve_py/`
4. Add tests in `tests/`, benchmarks in `tests/test_w_bench.py`
5. Update README.md feature table and CHANGELOG.md

## Documentation

Docs live in `docs/` on the `main` branch. The site is built with
**MkDocs + Material for MkDocs** and deployed to https://light-curve.snad.space/.

### Structure

```
docs/           — MkDocs source (Markdown + notebooks)
mkdocs.yml      — MkDocs config (nav, plugins, theme)
```

Key plugins: `mkdocstrings` (auto API docs from installed package), `mkdocs-jupyter`
(renders `.ipynb` notebooks as pages, executed at build time).

### Local dev server

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin
cd light-curve && maturin develop --extras=full --group=docs -q && cd ..
mkdocs serve          # live-reload at http://127.0.0.1:8000
```

Python-only doc changes need no Rust rebuild; re-run `maturin develop` only after Rust changes.

### Deploy

Pushing to `main` triggers `.github/workflows/docs.yml`, which deploys the `dev` version
to https://light-curve.snad.space/dev/. Pushing a `v*` tag deploys the `latest` version.

### PR checks

`test.yml` runs `mkdocs build --strict` on all PRs to catch broken links and missing references.

### Adding new features to docs

- Feature table: `docs/features/index.md` (manually maintained, grouped by category)
- API pages: `docs/features/api/<category>.md` — add `::: light_curve.FeatureName`
- Multiband features: `docs/features/multiband/api.md`

### Code block style

Every code block in the docs must be **self-contained**: include all imports and any
variable definitions needed to run it in isolation. Do not rely on variables defined in
a preceding block on the same page.
