# Changelog

All notable changes to `light-curve-python` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

— Pure-Python implemented features `FluxNNotDetBeforeFd` and `MagnitudeNNotDetBeforeFd`
- Pure-Python implemented `OtsuSplit.threshold` method

### Changed

- `rust-numpy` 0.15.0 -> 0.15.1
- `rand` 0.8.4 -> 0.8.5
- `enumflag2` 0.7.3 -> 0.7.4

### Deprecated

—

### Removed

—

### Fixed

—

### Security

—

## [0.5.4] 2021-12-20

### Added

- `ln_prior` submodule with `LnPrior1D` class and stand-alone functions to construct its instances
- `ln_prior` argument for `BazinFit` and `VillarFit` constructors which can be one of: `None`, `str` literals (currently the only useful value is 'hosseinzadeh2020' for `VillarFit`) or `list[LnPrior1D]`
- `Cargo.lock` is used to build the release packages and it is added to sdist, all these should make builds more reproducible

### Changed

- The project repository was split from Rust crates and moved into <https://gituhb.com/light-curve/light-curve-python>
- Maturin '>=0.12.15,<0.13' is required
- `light-curve-dmdt` version 0.5.0

## [0.5.3] 2021-12-16

### Added

- Python 3.10 support and binary wheels for supported platforms

### Changed

- Update `pyo3` to 0.15.1 and `rust-numpy` to 0.15.0

## [0.5.2] 2021-12-16

### Fixed

- Fix implementation of `OtsuSplit`, see [issue #150](https://github.com/light-curve/light-curve/issues/150)


## [0.5.1] 2021-12-15

### Added

- `init` and `bounds` arguments of `BazinFit` and `VillarFit` constructors

## [0.5.0] 2021-12-14

### Added

- `check: bool = True` keyword argument for `__call__` and `many` methods of feature classes. It coulb be used to check if input arrays are valid

### Changed

- `gsl` is a default Cargo feature now, which means that GSL must be installed to build this package by standard Python tools like `pip install`
- `light-curve-feature` 0.3 -> 0.4.1 transition brings MCMC improvements, changing feature names of `BazinFit` and significant changes of `VillarFit` feature set

### Removed

- `antifeatures` submodule of the Rust implementation

## [0.4.1] 2021-12-15

### Fixed

- Fix implementation of `OtsuSplit`, see [issue #150](https://github.com/light-curve/light-curve/issues/150)

## [0.4.0] 2021-11-26

### Added

- Pure-Python implemented `OtsuSplit` feature extractor (experimental)
- Python snippets in `README.md` are tested, this requires `pytest-markdown`
- A lot of new unit-tests for the pure-Python implementation
- New benchmarks to compare pure-Python and Rust implementations performance
- Publish Linux packages for `aarch64` and `ppc64le`

### Changed

- The Python package is renamed to `light-curve`, `light-curve-python` still exists as an alias
- Pure Python implementation of the most of the features are added, now Rust-implemented features live in `light_curve_ext` sub-package, while the Python implementation is in `light_curve_py`. Python-implemented feature extractors have an experimental status
- Now `dataclasses` (for Python 3.6 only) and `scipy` are required, they are used by the pure-Python implementation
- `maturin` version 0.12


### Fixed

- The Rust implemented classes `.__module__` was `buitins`, now it is `light_curve.light_curve_ext`

## [0.3.5] 2021-10-27

### Changed

- Rust edition 2021
- Minimum supported Rust version is 1.56
- Maturin version 0.11

## [0.3.4] 2021-10-18

### Fixed

- An exception shouldn't be raised for the case of small time-series length and non-`None` `fill_value`

## [0.3.3] 2021-10-14

### Added

- Support `dtype=np.float32` for feature extractors
- `_FeatureEvaluator.many(lcs)` method for parallel execution

## [0.3.2] 2021-08-30

### Changed

- Update `light-curve-feature` to `0.3.3`
- `__call__` docs for features

### Fixed

- `light-curve-feature` 0.3.3 fixes wrong `VillarFit` equation

## [0.3.1] 2021-08-16

### Added

- `mcmc_niter` and `lmsder_niter` parameters of `*Fit` features

### Changed

- Amd64 PyPI packages are manylinux2014
- PyPI releases compiled with `gsl` feature enabled

### Deprecated

- `gsl` will become a default feature in future releases

### Fixed

- `*Fit` `algorithm` parameter was marked as optional in the docstrings

## [0.3.0] 2021-08-10

### Added

- This `CHANGELOG.md` file
- `BazinFit` is enabled
- `VillarFit` feature

### Changed

- Remove `DmDt.point_from_columnar` and `DmDt.gausses_from_columnar`
- Update `ndarray` to 0.15.3 and `light-curve-dmdt` to 0.4.0
- Update `rust-numpy` and `pyo3` to 0.14
- Update `light-curve-feature` to 0.3.1
- Rename `nonlinear-fit` Cargo feature to `gsl`
- Docstrings improvements

## [0.2.x] pre 2021-05-31

—

## [0.1.x] pre 2020-09-09

—
