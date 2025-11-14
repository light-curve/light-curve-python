# GitHub Copilot Instructions for light-curve-python

## Project Overview

`light-curve-python` is a high-performance Python package for time-series feature extraction, particularly for astrophysical light curves. It's a hybrid Rust/Python project that wraps the [`light-curve-feature`](https://github.com/light-curve/light-curve-feature) and [`light-curve-dmdt`](https://github.com/light-curve/light-curve-dmdt) Rust crates.

### Key Technologies
- **Primary Language**: Rust (performance-critical feature extraction) + Python (API and experimental features)
- **Build System**: Maturin (Python-Rust bridge)
- **Python Support**: Python 3.9+ (including free-threaded 3.13t and 3.14t)
- **Testing**: pytest with benchmark support
- **CI/CD**: GitHub Actions with extensive platform coverage

## Repository Structure

```
light-curve-python/
├── .github/              # GitHub workflows and configuration
│   ├── workflows/        # CI/CD pipelines (test.yml, publish.yml)
│   └── dependabot.yml
├── light-curve/          # Main package directory
│   ├── src/              # Rust source code (PyO3 bindings)
│   ├── light_curve/      # Python package
│   │   ├── __init__.py
│   │   ├── light_curve_ext.py    # Rust implementation wrappers
│   │   └── light_curve_py/       # Pure Python implementations (experimental)
│   ├── tests/            # Python tests
│   ├── Cargo.toml        # Rust dependencies and configuration
│   ├── pyproject.toml    # Python package configuration
│   └── README.md         # Main documentation
└── light-curve-python/   # Alias package
```

## Development Workflow

### Setting Up Development Environment

1. **Prerequisites**:
   - Rust toolchain (stable, updated via `rustup`)
   - Python 3.9+
   - System dependencies: `libgsl-dev`, FFTW (on Ubuntu/Debian)
   - Optional: CMake and C++ compiler for Ceres solver support

2. **Initial Setup**:
   ```bash
   cd light-curve
   python3 -m venv venv
   source venv/bin/activate
   pip install maturin
   maturin develop --extras=dev
   ```

3. **Installing pre-commit hooks**:
   ```bash
   pre-commit install
   ```

### Building and Testing

#### Build Commands
- **Development build**: `maturin develop` (fast, unoptimized)
- **Release build**: `maturin develop --release` (slow, optimized)
- **Rebuild after Rust changes**: Run `maturin develop` again
- **Python-only changes**: No rebuild needed

#### Running Tests
- **All tests**: `pytest` or `python -m pytest`
- **With benchmarks**: `pytest --benchmark-enable`
- **Exclude slow benchmarks (recommended)**: `pytest -m "not (nobs or multi)" --benchmark-enable`
- **Specific test file**: `pytest tests/test_w_bench.py`

#### Linting and Formatting
- **Python**: `ruff format` and `ruff` (configured in pyproject.toml)
- **Rust**: `cargo fmt` and `cargo clippy`
- **Pre-commit**: Automatically runs on commit or via `pre-commit run --all-files`

### Code Style and Conventions

#### Python
- **Line length**: 120 characters
- **Target version**: Python 3.9+
- **Formatter**: Ruff format
- **Linter**: Ruff
- Follow existing patterns in `light_curve_py/` for experimental features

#### Rust
- **Edition**: 2024
- **MSRV**: 1.85 (check `Cargo.toml` for current version)
- **Formatter**: `rustfmt` (standard configuration)
- **Linter**: `clippy` with `-D warnings` (warnings as errors)
- Use PyO3 patterns for Python bindings in `src/`

### Cargo Features

The package has several compile-time features (see `Cargo.toml`):
- `abi3` (default): CPython ABI3 compatibility for stable Python ABI
- `ceres-source` (default): Ceres solver built from source
- `fftw-source` (default): FFTW built from source
- `fftw-mkl`: Intel MKL backend for FFTW (recommended for Intel CPUs)
- `gsl` (default): GNU Scientific Library support
- `mimalloc` (default): High-performance memory allocator

When suggesting build commands, consider: `--no-default-features --features=...` for minimal builds.

## Testing Requirements

### Test Coverage
- All new features must have tests
- Python tests live in `light-curve/tests/`
- Use `pytest-subtests` for parameterized tests
- Include docstring examples that can be tested with `markdown-pytest`

### Benchmark Tests
- Located in `tests/test_w_bench.py`
- Use `pytest-benchmark` for performance testing
- Mark slow benchmarks with `@pytest.mark.nobs` or `@pytest.mark.multi`

### Documentation Tests
- README examples are tested via `markdown-pytest`
- Use `<!-- name: test_name -->` comments before code blocks to enable testing

## Important Considerations

### Performance
- Rust implementation is 1.5-50× faster than pure Python
- Use Rust for performance-critical features
- Python implementation (`light_curve_py`) is for experimental features and testing
- Consider multithreading via `.many()` method for batch processing

### Compatibility
- Support Python 3.9-3.14 (including free-threaded variants)
- Maintain backward compatibility
- Test on multiple platforms: Linux (x86_64, aarch64), macOS, Windows
- ABI3 wheels ensure forward compatibility with future Python versions

### Dependencies
- **Rust dependencies**: Managed via `Cargo.toml`, keep `Cargo.lock` committed
- **Python dependencies**: Minimal required (`numpy`), optional extras (`full`, `test`, `dev`)
- System dependencies required: GSL (optional but default)

## Common Tasks

### Adding a New Feature Extractor

1. **Rust implementation**:
   - Typically implemented in the upstream `light-curve-feature` crate
   - Add Python bindings in `src/` if needed
   - Export in `src/lib.rs`

2. **Python wrapper**:
   - Import in `light_curve/__init__.py`
   - Add experimental Python version in `light_curve_py/` if needed

3. **Tests**:
   - Add unit tests in `tests/`
   - Add benchmarks in `tests/test_w_bench.py`
   - Update README with feature documentation

4. **Documentation**:
   - Update README.md feature table
   - Add usage examples
   - Update CHANGELOG.md

### Updating Dependencies

- **Rust**: Update `Cargo.toml`, test locally, update CI if needed
- **Python**: Update `pyproject.toml`, ensure compatibility
- **System**: Update CI workflows if system dependencies change

### Debugging Build Issues

- Check system dependencies (GSL, CMake, C++ compiler)
- Verify Rust toolchain version meets MSRV
- For Ceres/FFTW issues: try `--no-default-features --features=fftw-source`
- Enable verbose output: `maturin develop -v`

## CI/CD Pipeline

### GitHub Actions Workflows

- **test.yml**: Main test suite
  - Tests on Ubuntu (x86_64, aarch64) with Python 3.9-3.14
  - Runs `cargo fmt`, `cargo clippy`, and coverage
  - Benchmarks with CodSpeed
  - MSRV verification

- **publish.yml**: Package distribution
  - Builds wheels for multiple platforms/architectures
  - Publishes to PyPI and TestPyPI
  - Uses `cibuildwheel` for cross-platform builds

### Working with CI

- All PRs trigger the test workflow
- Fix linting issues locally before pushing: `pre-commit run --all-files`
- For CI failures, check job logs for specific errors
- Coverage reports go to Codecov

## Resources

- **Main documentation**: `light-curve/README.md`
- **Rust crate docs**: https://docs.rs/light-curve-feature/
- **Issue tracker**: https://github.com/light-curve/light-curve-python/issues
- **PyPI**: https://pypi.org/project/light-curve/
- **Conda**: https://anaconda.org/conda-forge/light-curve-python

## Best Practices for Contributors

1. **Minimal changes**: Make surgical, focused modifications
2. **Test early**: Run tests after each significant change
3. **Performance matters**: Profile before optimizing, prefer Rust for hot paths
4. **Documentation**: Keep README and docstrings up-to-date
5. **Changelog**: Update CHANGELOG.md for user-facing changes
6. **Compatibility**: Ensure changes work across supported Python versions
7. **Pre-commit**: Use pre-commit hooks to catch issues early
8. **Follow patterns**: Match existing code style and structure
