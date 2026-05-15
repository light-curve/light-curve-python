# Contributing

## Environment setup

Requires Rust ≥ 1.85 ([rustup](https://rustup.rs/)) and Python ≥ 3.10.

```bash
git clone https://github.com/light-curve/light-curve-python.git
cd light-curve-python/light-curve
python3 -m venv venv && source venv/bin/activate
pip install maturin
maturin develop --group dev   # builds Rust extension + installs dev deps
pre-commit install
```

After this, re-activate the venv and run `maturin develop` after Rust changes.
Python-only changes need no rebuild.

## Tests

```bash
pytest                                # all tests
pytest --benchmark-enable             # include benchmarks
pytest tests/light_curve_ext/test_feature.py   # single file
```

Tests also run against the README examples via `markdown-pytest`.

## Linting

```bash
ruff check .            # Python lint
ruff format .           # Python format
cargo fmt --manifest-path=Cargo.toml
cargo clippy --all-targets -- -D warnings
# or all at once:
pre-commit run --all-files
```

## Cargo features

| Feature | Default | Effect |
|---------|---------|--------|
| `abi3` | ✓ | Stable CPython ABI; disable for PyPy or free-threaded builds |
| `ceres-source` | ✓ | Build Ceres solver from source (requires C++ compiler + CMake) |
| `ceres-system` | | Link against system-installed Ceres instead |
| `gsl` | ✓ | Enable GSL backend for `BazinFit`/`VillarFit` (requires `libgsl-dev`) |
| `mkl` | | Intel MKL FFTW backend for fast periodogram (Intel CPUs) |
| `mimalloc` | ✓ | mimalloc allocator — up to 2× speedup for cheap features |

Pass features with `--features` / `--no-default-features`:

```bash
maturin develop --release --no-default-features --features=abi3,ceres-source
```

## Adding a new feature extractor

1. Implement the feature in the upstream [`light-curve-feature`](https://github.com/light-curve/light-curve) Rust crate.
2. Add PyO3 bindings in `src/features.rs`, export in `src/lib.rs`.
3. Optionally add a pure-Python version in `light_curve/light_curve_py/`.
4. Add tests in `tests/`, benchmarks in `tests/test_w_bench.py`.
5. Update `README.md` feature table and `CHANGELOG.md`.

## Releases

See the CI publish workflow. Releases are triggered by pushing a version tag.
