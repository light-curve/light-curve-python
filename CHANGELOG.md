# Changelog

All notable changes to `light-curve-python` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

--

### Changed

- **Breaking change in experimental features** Multiband support is introduced for features implemented in Python. It changes class inheritance interface in a backward-incompatible way
- **Breaking change in experimental features** All experimental features implemented in Python require keyword-only arguments in their constructors. Also, all names of the arguments are changed to be the same as for Rust features
- Bump `pyO3` 0.18.3 -> 0.19.1, it simplified signature generations for classes https://github.com/light-curve/light-curve-python/pull/230
- Bump `rust-numpy` 0.18.0 -> 0.19.0 https://github.com/light-curve/light-curve-python/pull/230

### Deprecated

--

### Removed

--

### Fixed

--

### Security

--

## [0.7.3] 2023-07-06

### Added

- All Rust features got `.to_json()` method to serialize their state to JSON string. https://github.com/light-curve/light-curve-python/pull/226
- New special Rust feature `JSONDeserializedFeature` and a helper function `feature_from_json()` to deserialize features from JSON string. https://github.com/light-curve/light-curve-python/pull/226
- Build: "abi3" and "abi3-py310" Cargo features (the least one is enabled by the first one) to build a wheel for CPython 3.10+. This stable ABI wheel is less performant than the regular one, but it is compatible with all future Python versions. See [PEP 384](https://www.python.org/dev/peps/pep-0384/) for details. https://github.com/light-curve/light-curve-python/issues/79

### Changed

- **Build breaking**: the only Python build requirement `maturin` updated from v0.14.x to v1.0 https://github.com/light-curve/light-curve-python/pull/216 https://github.com/light-curve/light-curve-python/pull/215
- CI: bump cibuildwheel to 2.13.1 https://github.com/light-curve/light-curve-python/pull/225
- Bump `itertools` 0.10.5 -> 0.11.0 https://github.com/light-curve/light-curve-python/pull/224
- Bump `pyO3` 0.18.2 -> 0.18.3 https://github.com/light-curve/light-curve-python/pull/207

### Fixed

- Building from sdist on x86-64 macOS required manual setting of `$MACOSX_DEPLOYMENT_TARGET` to 10.9 or higher. Recent `maturin` update allowed us to specify it via `pyproject.toml`


## [0.7.2] 2023-04-12

### Added

- Feature transformations via `transform` constructor keyword. For most of the features it could accept string with a transformation name such as 'arcsinh' or 'clipped_lg', `True` or 'default' for the default transformation, `None` or `False` for no transformation https://github.com/light-curve/light-curve-python/issues/184 https://github.com/light-curve/light-curve-python/pull/188
- Binary wheels for x86_64 Windows built with no Ceres nor GSL features https://github.com/light-curve/light-curve-python/issues/12 https://github.com/light-curve/light-curve-python/pull/185
- `enum-iterator` crate dependency https://github.com/light-curve/light-curve-python/pull/188
- CI: code coverage with `codecov` https://github.com/light-curve/light-curve-python/pull/197
- Development: now project has extras for testing (`test`) and development (`dev`) https://github.com/light-curve/light-curve-python/pull/197

### Changed

- **Build breaking**: minimum supported Rust version (MSRV) is bump from 1.60 to 1.62
- Bump `light-curve-feature` 0.5.4 -> 0.5.5
- Bump `pyO3` 0.18.1 -> 0.18.2
- Most of the parametric features have default values for their parameters now, which, due to `pyO3` limitations, are not presented in the signatures, but documented in the docstrings. It also makes Python and Rust implementations more consistent https://github.com/light-curve/light-curve-python/issues/194 https://github.com/light-curve/light-curve-python/pull/195
- Development: switch from `pytest-markdown` to `markdown-pytest` which allowed us to use up-to-date pytest https://github.com/light-curve/light-curve-python/pull/198

### Deprecated

- `BazinFit` and `VillarFit` constructors will not accept `None` for `mcmc_niter`, `ceres_niter`, and `lmsder_niter` arguments in the future, just do not specify them to use defaults instead. https://github.com/light-curve/light-curve-python/pull/195
- `Periodogram` constructor will not accept `None` for `peaks`, `resolution`, `max_freq_factor`, `nyquist` and `fast` in the future, just do not specify them to use defaults instead. https://github.com/light-curve/light-curve-python/pull/195

### Fixed

- `Bins` feature had non-optimal lower boundary check for time series length: it checked if it is at least unity for any underlying features. Now it takes underlying feature requirements into account. It was fixed by updating `light-curve-feature` to v0.5.5.


## [0.7.1] 2023-03-17

### Fixed

- Bug introduced in v0.6.5: `*Fit.model(t, params)` wrongly checked `t` and `params` arrays to have the same length


## [0.7.0] 2023-03-16

### Added

- `BazinFit` and `VillarFit` have got `ceres` and `mcmc-ceres` algorithms using [Ceres Solver](http://ceres-solver.org) as a non-linear least squares optimizer. `ceres` is found to be more robust than `lmsder` algorithm (available via `gsl` Cargo feature) but working roughly twice slower. Ceres can be built from source (`ceres-source` Cargo feature, enabled by default in `Cargo.toml`) or linked to system library (`ceres-system` Cargo feature, enabled for cibuildwheel in `pyproject.toml`)

### Changed

- **API breaking:** Features' `__call__()` signature changed to make `sorted=None`, `check=True` and `fill_value=None` arguments to be keyword-only
- **API breaking:** Features' `many()` signature changed to make all arguments but the first `lcs` to be keyword-only
- **API breaking:** `Bins` constructor signature changed to make `offset` and `window` arguments to be keyword-only. For Rust implementation `__getnewargs__` is replaced with `__getnewargs_ex__`. Please note that for the specific case of Python implementation and Python version < 3.10, `Bins` still accepts positional arguments
- **API breaking:** `BazinFit` and `VillarFit` constructor signatures changed to make everything but the first `lcs` argument to be keyword-only
- **API breaking:** `Periodogram` constructor signature changed to make all arguments to be keyword-only
- **API breaking:** `DmDt` constructor signature changed to make all arguments but `dt` and `dm` to be keyword-only, `__getnewargs__` is replaced with `__getnewargs_ex__`. `DmDt.from_borders` class-method constructor has all arguments to be keyword-only
- **API breaking:** `DmDt` methods' signatures changed to make all arguments but data (like `t`, `t, m` or `lcs`) to be keyword-only
- **Build breaking:** building with Ceres Solver (`ceres-source` Cargo feature) is now a default, and potentially could break a building pipeline in some cases. If you want to build without Ceres Solver, you need to explicitly disable default features with `--no-default-features` maturin flag
- CI: switch from `macos-11` to `macos-latest` for testing
- Bump `pyo3` 0.17.3 -> 0.18.1
- Bump `rust-numpy` 0.17.2 -> 0.18.0

### Removed

- **Build breaking:** `fftw-static`, `fftw-dynamic`, `mkl` Cargo features are removed after deprecation in v0.6.2 and replaced with `fftw-source`, `fftw-system` and `fftw-mkl`.


## [0.6.6] 2023-03-17

### Fixed

- Bug introduced in v0.6.5: `*Fit.model(t, params)` wrongly checked `t` and `params` arrays to have the same length


## [0.6.5] 2023-02-22

### Fixed

- Reduce Rust-Python inter-op cost for numpy arrays significantly. It dropped from ~4 μs per array to ~100ns. https://github.com/light-curve/light-curve-python/pull/174


## [0.6.4] 2023-01-19

### Added

- Initial `copy` and `pickle` (minimum protocol version is 2) support for feature extractors and `DmDt` https://github.com/light-curve/light-curve-python/issues/103 https://github.com/light-curve/light-curve-python/pull/145 https://github.com/light-curve/light-curve-python/pull/150
- `serde` v1 and `serde-pickle` v1 Rust dependencies. `serde-pickle` is an arbitrary choice of a (de)serialization binary format, but it could be useful in the future having a way to inspect Rust structures from Python https://github.com/light-curve/light-curve-python/pull/145

### Changed

- Build environment: minimum support Rust version (MSRV) is bumped 1.57 -> 1.60
- Bump `light-curve-dmdt` 0.6.0 -> 0.7.1

### Fixed

- `BazinFit` and `VillarFit` docs are clarified for `.model()` and `ln_prior` usage https://github.com/light-curve/light-curve-python/issues/125 https://github.com/light-curve/light-curve-python/pull/146

## [0.6.3] 2022-12-23

No changes, it was accidentally released instead of `0.6.2`

## [0.6.2] 2022-12-27

- `OtsuSplit` implementation in Rust https://github.com/light-curve/light-curve-python/issues/120 https://github.com/light-curve/light-curve-python/pull/123

### Changed

- `light-curve-feature` 0.5.0 -> 0.5.2 https://github.com/light-curve/light-curve-python/pull/123
- `light-curve-dmdt` 0.5.0 -> 0.6.0
- `pyO3` 0.16.6 -> 0.17.3
- `rust-numpy` 0.16.2 -> 0.17.2
- CI: binary wheels are now built using our custom [manylinux/musllinux images](https://github.com/light-curve/base-docker-images), which include FFTW library optimised to use platform-specific SIMD instructions. It should give up to 50% performance gain for `Periodogram` at all Linux platforms but `x86_64` where we use MKL https://github.com/light-curve/light-curve-python/pull/134
- We don't provide binary wheels for Linux i686 anymore, please contact us if you need them https://github.com/light-curve/light-curve-python/pull/134
- wheel build dependency: `maturin` 0.13.x -> 0.14.x https://github.com/light-curve/light-curve-python/issues/130 https://github.com/light-curve/light-curve-python/pull/135

### Deprecated

- cargo features "fftw-dynamic", "fftw-static" and "mkl" are renamed to "fftw-system", "fftw-source" and "fftw-mkl" correspondingly https://github.com/light-curve/light-curve-python/pull/137

### Fixed

— fix `threshold` method in Python according to Rust implementation https://github.com/light-curve/light-curve-python/pull/123

## [0.6.1] 2022-11-01

### Added

- CI: Python 3.11 binary wheels

### Changed

- `const_format` 0.2.25 -> 0.2.30
- `itertools` 0.10.3 -> 0.10.5
- `pyo3` 0.16.5 -> 0.16.6
- `thiserror` 1.0.31 -> 1.0.37
- wheel build dependency: `maturin` 0.12.x -> 0.13.x
- CI: `macos-10.15` GitHub actions runners are EOL, we switched to `macos-11`
- CI: `cibuildwheel` 2.10.2 -> 2.11.2

### Deprecated

—

### Removed

—

### Fixed

- https://github.com/light-curve/light-curve-python/issues/110 Experimental warning message specifies a class name https://github.com/light-curve/light-curve-python/pull/111
- Support of `setuptools` v61+

### Security

—

## [0.6.0] 2022-07-04

### Changed

- **Breaking:** drop Python 3.6 support
- **Breaking:** minimum supported Rust version (MSRV) 1.56 -> 1.57
- `light-curve-feature` 0.4.6 -> 0.5.0, causes MSRV change
- `rust-numpy` 0.15.1 -> 0.16.2
- `py03` 0.15.1 -> 0.16.3

## [0.5.9] 2022-06-15

### Added

- `macro_const` 0.1.0 dependency

### Fixed

- docstring for `many` method is added to each Rust feature class


## [0.5.8] 2022-06-14

### Changed

- `light-curve-feature` 0.4.5 -> 0.4.6

### Fixed

- Minimum supported Rust version 1.56 wasn't actually supported, `light-curve-feature` upgrade fixed it


## [0.5.7] 2022-06-10

### Added

- CI: build Windows w/o GSL

### Changed

- `light-curve-feature` 0.4.4 -> 0.4.5

### Fixed

- `light-curve-feature` update fixes `NaN value of lnprob` problem with MCMC fit in `{Bazin,Villar}Fit`

## [0.5.6] 2022-06-06

### Changed

- Infrastructure: [cibuildweel](https://cibuildwheel.readthedocs.io/en/stable/) for package publishing
- `light-curve-feature` 0.4.1 -> 0.4.4
- `thiserror` 1.0.24 -> 1.0.31
- `enumflags2` 0.7.4 -> 0.7.5
- `rayon` 1.5.1 -> 1.5.3
- `pyo3` 0.15.1 -> 0.15.2
- `const_format` 0.2.22 -> 0.2.24

### Fixed

- `light-curve-feature` update fixes `{Bazin,Villar}Fit` overflow panic
- Benchmark and test for `MagnitudePercentageRatio`

## [0.5.5] 2022-04-05

### Added

- Pure-Python implemented features `FluxNNotDetBeforeFd` and `MagnitudeNNotDetBeforeFd`
- Pure-Python implemented `OtsuSplit.threshold` method

### Changed

- `rust-numpy` 0.15.0 -> 0.15.1
- `rand` 0.8.4 -> 0.8.5
- `enumflag2` 0.7.3 -> 0.7.4

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
