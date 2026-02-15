# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`light-curve-python` is a hybrid Rust/Python package for time-series feature extraction in astrophysics. It wraps the
`light-curve-feature` and `light-curve-dmdt` Rust crates via PyO3/Maturin bindings.

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
maturin develop --extras=dev          # Dev build with test dependencies
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

**Feature extraction pattern**: Each feature (e.g., `Amplitude`, `BazinFit`, `Periodogram`) is a class with
`__call__(t, m, sigma, ...)` for single light curves and `.many(...)` for batch processing with reduced Python-Rust
overhead.

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
3. Optionally add experimental Python version in `light_curve_py/`
4. Add tests in `tests/`, benchmarks in `tests/test_w_bench.py`
5. Update README.md feature table and CHANGELOG.md
