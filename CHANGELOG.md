# Changelog

All notable changes to `light-curve-python` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

--

### Changed

--

### Deprecated

--

### Removed

--

### Fixed

- A problem with pickling of `Periodogram` which caused wrong results from `.power` and `.freq_power` for a deserialized
  object https://github.com/light-curve/light-curve-python/pull/532

### Security

--

## [0.10.4] 2025-06-11

### Added

- Periodogram(freqs: ArrayLike | None = None) is added to set fixed user-defined frequency
  grids https://github.com/light-curve/light-curve-python/pull/528
- `Periodogram.power` method https://github.com/light-curve/light-curve-python/pull/529

### Changed

- Bump `light-curve-feature` to 0.10.0

## [0.10.3] 2025-05-22

### Added

- Mark the module as no-GIL, which enables free-threaded Python (can be built from source, not provided so far via
  PyPI/conda) https://github.com/light-curve/light-curve-python/pull/499
- Allow non-numpy inputs and casting mismatched f32 arrays to f64 for the feature extractions with newly added
  `cast: bool = False` argument. We plan to change the default value to `True` in a future 0.x version.
  https://github.com/light-curve/light-curve-python/issues/509 https://github.com/light-curve/light-curve-python/pull/512

### Changed

- **PyPI wheels change**: bump Musl PyPI wheels compatibility from musllinux 1.1 to
  1.2 https://github.com/light-curve/light-curve-python/pull/503
- **Build breaking**: minimum supported Rust version (MSRV) is changed from 1.67 to
  1.85 https://github.com/light-curve/light-curve-python/pull/516
- We also migrated from Rust edition 2021 to 2024 https://github.com/light-curve/light-curve-python/pull/516
- Bump both `PyO3` and `rust-numpy` to
  v0.25 https://github.com/light-curve/light-curve-python/pull/499 https://github.com/light-curve/light-curve-python/pull/520
- Bump `light-curve-feature` to v0.9.0

### Fixed

- Fix error messages for invalid
  inputs https://github.com/light-curve/light-curve-python/issues/505 https://github.com/light-curve/light-curve-python/pull/510

## [0.10.2] 2025-03-25

### Fixed

- Restored minimum supported Rust version (MSRV) compatibility for v1.67+

## [0.10.1] 2025-03-25 (DELETED)

Release is deleted from PyPI and has never been released on conda-forge.
The reason is a bug fixed in 0.10.2.

### Changed

- **Experimental Feature Breaking**: change parameter limits for
  Rainbow https://github.com/light-curve/light-curve-python/pull/494

### Removed

- We stop building PPC64le wheels in CI and publishing them to
  PyPI https://github.com/light-curve/light-curve-python/issues/479 https://github.com/light-curve/light-curve-python/pull/480

### Fixed

- Rainbow multi-band scaler didn't work with list
  inputs https://github.com/light-curve/light-curve-python/issues/492 https://github.com/light-curve/light-curve-python/pull/493

## [0.10.0] 2025-01-07

### Changed

- **Breaking** macOS x86\_64 binary wheel now requires macOS 13 instead of
  12 https://github.com/light-curve/light-curve-python/issues/437 https://github.com/light-curve/light-curve-python/pull/446
- **Breaking** Minimum supported Python version is 3.9 due to `rust-numpy` 0.23 requirement
- Default ABI3 version is also bumped to 3.9
- Bump `pyO3` to 0.23, which should potentially support
  free-threading, but `rust-numpy` still doesn't https://github.com/light-curve/light-curve-python/pull/457
- Bump `rust-numpy` to 0.23, should potentially bring a better `numpy` v2
  support https://github.com/light-curve/light-curve-python/pull/457
- Bump `ndarray` to 0.16 https://github.com/light-curve/light-curve-python/pull/458
- Bump `light-curve-feature` to 0.8.0 https://github.com/light-curve/light-curve-python/pull/458
- Bump `light-curve-dmdt` to 0.8.0 https://github.com/light-curve/light-curve-python/pull/458

### Fixed

- `pyproject.toml`: move `tool.setuptools.dynamic` to `project.dynamic` which fixes build with `maturin`
  v1.8.x https://github.com/light-curve/light-curve-python/pull/467

## [0.9.6] 2024-10-01

### Changed

- `LinexpBolometricTerm` for `Rainbow` changed to clip negative values to
  zero https://github.com/light-curve/light-curve-python/pull/430

## [0.9.5] 2024-09-20

### Added

- More variants of temperature and bollometric functions for
  `RainbowFit` https://github.com/light-curve/light-curve-python/pull/411

### Changed

- Change boundary conditions for `RainbowFit` https://github.com/light-curve/light-curve-python/pull/414

### Fixed

- Package import doesn't fail if `scipy` is missed https://github.com/light-curve/light-curve-python/pull/422

## [0.9.4] 2024-09-11

### Changed

- Experimental Rainbow fit features use maximum likelihood cost function instead of least
  squares https://github.com/light-curve/light-curve-python/pull/407

## [0.9.3] 2024-06-17

### Fixed

- Small bug fix in MagnitudeNNotDetBeforeFd for numpy v2.0
  compatibility https://github.com/light-curve/light-curve-python/pull/383

## [0.9.2] 2024-06-05

### Added

- `Roms` feature implemented in
  Rust https://github.com/light-curve/light-curve-python/pull/362 https://github.com/light-curve/light-curve-python/issues/356

### Fixed

- PyPI wheels used to have wrong platform tags, actual minimum macOS versions are those used at CI: 12.0+ for x86_64 and
  14.0+ for
  arm64. https://github.com/light-curve/light-curve-python/issues/376 https://github.com/light-curve/light-curve-python/pull/378

## [0.9.1] 2024-04-23

### Added

- `PeakToPeakVar` experimental feature https://github.com/light-curve/light-curve-python/pull/332

### Changed

- Bump `pyO3` to 0.21
- Bump `rust-numpy` to 0.21
- "abi3" is a default Cargo feature now

### Removed

- Stop publishing PyPy wheels to PyPI. We still publish all CPython wheels we published
  previously https://github.com/light-curve/light-curve-python/issues/345 https://github.com/light-curve/light-curve-python/pull/347

### Fixed

- Bug which prevents initialization of Rust's `*Fit` features if compiled without Ceres or GSL (our PyPi Windows
  wheels) https://github.com/light-curve/light-curve-python/issues/343 https://github.com/light-curve/light-curve-python/pull/344

## [0.9.0] 2024-03-06

### Added

- `Roms` (robust median statistics) experimental feature, a variability index based on the deviation of observations
  from the median. https://github.com/light-curve/light-curve-python/pull/315 Thanks @GaluTi for their first
  contribution

### Changed

- **Breaking** `RainbowFit` is significantly refactored to make it more flexible for usage of different bolometric and
  temperature functions. It deletes `RainbowRisingFit` and `RainbowSymmetricFit` in favor of a single `RainbowFit`
  feature. https://github.com/light-curve/light-curve-python/pull/324
- **Breaking**: stop supporting Python 3.7 https://github.com/light-curve/light-curve-python/pull/282
- We stop distributing CPython wheels for specific Python versions, now we distribute ABI3 wheels which are compatible
  with all future Python versions

### Deprecated

--

### Removed

- **Build breaking**: `abi311` Cargo feature is removed, now we plan to have `abi3` feature only, which would correspond
  to the minimum supported Python version. Feel free to use `pyo3/abi3..` features directly for newer ABI versions.

### Fixed

--

### Security

--

## [0.8.2] 2024-02-27

### Added

- New flavour of `RainbowFit`: `RainbowSymmetricFit` which will replace both `RainbowFit` and `RainbowRisingFit` in the
  future. https://github.com/light-curve/light-curve-python/pull/314 Thanks @karpov-sv for their first contribution
- New cargo build-time feature, `mimalloc`, it is default feature now. When activated, it gives up to 2.9x of
  performance boost for some "fast" features. https://github.com/light-curve/light-curve-python/pull/302

### Changed

- Refactoring of rainbow features, it reduces code duplication and makes it easier to add new variants like `RainbowFit`
  and `RainbowRisingFit` in the future https://github.com/light-curve/light-curve-python/pull/293
- Another change for `Rainbow` features is changing `minuit` optimization
  parameters https://github.com/light-curve/light-curve-python/pull/314
- **Build breaking**: bump `light-curve-feature` to v0.7.0, which requires ceres-solver v2.2 for `ceres-system` Cargo
  feature.

### Fixed

- `RainbowFit` and `RainbowRisingFit` initial guesses for baseline fluxes are now consistent with limits. We also use
  band information to make initial guesses and limits more accurate. Note, that this change leads to different results
  comparing to previous versions. https://github.com/light-curve/light-curve-python/pull/293

## [0.8.1] 2023-11-30

### Added

- `RainbowRisingFit` experimental feature, https://github.com/light-curve/light-curve-python/pull/278
  by [@erusseil](https://github.com/erusseil)

## [0.8.0] 2023-09-20

### Added

- **Breaking change in experimental features**: Multiband support is introduced for features implemented in Python. It
  changes class inheritance interface in a backward-incompatible way
- `light-curve[full]` extras which installs all optional Python dependencies required by experimental features
- New `LinexpFit` feature for parametric model fit comes with `light-curve-feature` v0.6.0
- Experimental `RainbowFit` feature for fitting multiband light curves with a single model, Russeil+23 in prep. It
  requires Python 3.8 or later because of `iminuit` dependency
- Optional `iminuit>=2,<3` Python dependency (included into `[full]`) for `RainbowFit` feature
- Add `once_cell` v1 dependency

### Changed

- **Breaking change in experimental features** `scipy` dependency is now optional for experimental features implemented
  in Python
- **Breaking change in experimental features**: All experimental features implemented in Python require keyword-only
  arguments in their constructors. Also, all names of the arguments are changed to be the same as for Rust features
- **Build breaking**: "abi3-py310" Cargo feature is replaced with "abi3-py311". "abi3" feature is now linked to "
  abi3-py311" feature. This is because our aim with ABI is to support future versions of Python
- **Build breaking**: minimum supported Rust version (MSRV) is changed from 1.62 to 1.67 (released 2023-01-26)
- Update `*Fit` fatures doc-strings to list names of the features they output
- Bump `light-curve-feature` 0.5.5 -> 0.6.0
- Bump `pyO3` 0.18.3 -> 0.19.1, it simplified signature generations for
  classes https://github.com/light-curve/light-curve-python/pull/230
- Bump `rust-numpy` 0.18.0 -> 0.19.0 https://github.com/light-curve/light-curve-python/pull/230
- Bump `enum-iterator` 1.2.0 -> 1.4.1 https://github.com/light-curve/light-curve-python/pull/233
- Bump `thiserror` 1.0.41 -> 1.0.48 https://github.com/light-curve/light-curve-python/pull/242

## [0.7.3] 2023-07-06

### Added

- All Rust features got `.to_json()` method to serialize their state to JSON
  string. https://github.com/light-curve/light-curve-python/pull/226
- New special Rust feature `JSONDeserializedFeature` and a helper function `feature_from_json()` to deserialize features
  from JSON string. https://github.com/light-curve/light-curve-python/pull/226
- Build: "abi3" and "abi3-py310" Cargo features (the least one is enabled by the first one) to build a wheel for CPython
  3.10+. This stable ABI wheel is less performant than the regular one, but it is compatible with all future Python
  versions. See [PEP 384](https://www.python.org/dev/peps/pep-0384/) for
  details. https://github.com/light-curve/light-curve-python/issues/79

### Changed

- **Build breaking**: the only Python build requirement `maturin` updated from v0.14.x to
  v1.0 https://github.com/light-curve/light-curve-python/pull/216 https://github.com/light-curve/light-curve-python/pull/215
- CI: bump cibuildwheel to 2.13.1 https://github.com/light-curve/light-curve-python/pull/225
- Bump `itertools` 0.10.5 -> 0.11.0 https://github.com/light-curve/light-curve-python/pull/224
- Bump `pyO3` 0.18.2 -> 0.18.3 https://github.com/light-curve/light-curve-python/pull/207

### Fixed

- Building from sdist on x86-64 macOS required manual setting of `$MACOSX_DEPLOYMENT_TARGET` to 10.9 or higher.
  Recent `maturin` update allowed us to specify it via `pyproject.toml`

## [0.7.2] 2023-04-12

### Added

- Feature transformations via `transform` constructor keyword. For most of the features it could accept string with a
  transformation name such as 'arcsinh' or 'clipped_lg', `True` or 'default' for the default transformation, `None`
  or `False` for no
  transformation https://github.com/light-curve/light-curve-python/issues/184 https://github.com/light-curve/light-curve-python/pull/188
- Binary wheels for x86_64 Windows built with no Ceres nor GSL
  features https://github.com/light-curve/light-curve-python/issues/12 https://github.com/light-curve/light-curve-python/pull/185
- `enum-iterator` crate dependency https://github.com/light-curve/light-curve-python/pull/188
- CI: code coverage with `codecov` https://github.com/light-curve/light-curve-python/pull/197
- Development: now project has extras for testing (`test`) and
  development (`dev`) https://github.com/light-curve/light-curve-python/pull/197

### Changed

- **Build breaking**: minimum supported Rust version (MSRV) is bump from 1.60 to 1.62
- Bump `light-curve-feature` 0.5.4 -> 0.5.5
- Bump `pyO3` 0.18.1 -> 0.18.2
- Most of the parametric features have default values for their parameters now, which, due to `pyO3` limitations, are
  not presented in the signatures, but documented in the docstrings. It also makes Python and Rust implementations more
  consistent https://github.com/light-curve/light-curve-python/issues/194 https://github.com/light-curve/light-curve-python/pull/195
- Development: switch from `pytest-markdown` to `markdown-pytest` which allowed us to use up-to-date
  pytest https://github.com/light-curve/light-curve-python/pull/198

### Deprecated

- `BazinFit` and `VillarFit` constructors will not accept `None` for `mcmc_niter`, `ceres_niter`, and `lmsder_niter`
  arguments in the future, just do not specify them to use defaults
  instead. https://github.com/light-curve/light-curve-python/pull/195
- `Periodogram` constructor will not accept `None` for `peaks`, `resolution`, `max_freq_factor`, `nyquist` and `fast` in
  the future, just do not specify them to use defaults
  instead. https://github.com/light-curve/light-curve-python/pull/195

### Fixed

- `Bins` feature had non-optimal lower boundary check for time series length: it checked if it is at least unity for any
  underlying features. Now it takes underlying feature requirements into account. It was fixed by
  updating `light-curve-feature` to v0.5.5.

## [0.7.1] 2023-03-17

### Fixed

- Bug introduced in v0.6.5: `*Fit.model(t, params)` wrongly checked `t` and `params` arrays to have the same length

## [0.7.0] 2023-03-16

### Added

- `BazinFit` and `VillarFit` have got `ceres` and `mcmc-ceres` algorithms using [Ceres Solver](http://ceres-solver.org)
  as a non-linear least squares optimizer. `ceres` is found to be more robust than `lmsder` algorithm (available
  via `gsl` Cargo feature) but working roughly twice slower. Ceres can be built from source (`ceres-source` Cargo
  feature, enabled by default in `Cargo.toml`) or linked to system library (`ceres-system` Cargo feature, enabled for
  cibuildwheel in `pyproject.toml`)

### Changed

- **API breaking:** Features' `__call__()` signature changed to make `sorted=None`, `check=True` and `fill_value=None`
  arguments to be keyword-only
- **API breaking:** Features' `many()` signature changed to make all arguments but the first `lcs` to be keyword-only
- **API breaking:** `Bins` constructor signature changed to make `offset` and `window` arguments to be keyword-only. For
  Rust implementation `__getnewargs__` is replaced with `__getnewargs_ex__`. Please note that for the specific case of
  Python implementation and Python version < 3.10, `Bins` still accepts positional arguments
- **API breaking:** `BazinFit` and `VillarFit` constructor signatures changed to make everything but the first `lcs`
  argument to be keyword-only
- **API breaking:** `Periodogram` constructor signature changed to make all arguments to be keyword-only
- **API breaking:** `DmDt` constructor signature changed to make all arguments but `dt` and `dm` to be
  keyword-only, `__getnewargs__` is replaced with `__getnewargs_ex__`. `DmDt.from_borders` class-method constructor has
  all arguments to be keyword-only
- **API breaking:** `DmDt` methods' signatures changed to make all arguments but data (like `t`, `t, m` or `lcs`) to be
  keyword-only
- **Build breaking:** building with Ceres Solver (`ceres-source` Cargo feature) is now a default, and potentially could
  break a building pipeline in some cases. If you want to build without Ceres Solver, you need to explicitly disable
  default features with `--no-default-features` maturin flag
- CI: switch from `macos-11` to `macos-latest` for testing
- Bump `pyo3` 0.17.3 -> 0.18.1
- Bump `rust-numpy` 0.17.2 -> 0.18.0

### Removed

- **Build breaking:** `fftw-static`, `fftw-dynamic`, `mkl` Cargo features are removed after deprecation in v0.6.2 and
  replaced with `fftw-source`, `fftw-system` and `fftw-mkl`.

## [0.6.6] 2023-03-17

### Fixed

- Bug introduced in v0.6.5: `*Fit.model(t, params)` wrongly checked `t` and `params` arrays to have the same length

## [0.6.5] 2023-02-22

### Fixed

- Reduce Rust-Python inter-op cost for numpy arrays significantly. It dropped from ~4 μs per array to ~
  100ns. https://github.com/light-curve/light-curve-python/pull/174

## [0.6.4] 2023-01-19

### Added

- Initial `copy` and `pickle` (minimum protocol version is 2) support for feature extractors
  and
  `DmDt` https://github.com/light-curve/light-curve-python/issues/103 https://github.com/light-curve/light-curve-python/pull/145 https://github.com/light-curve/light-curve-python/pull/150
- `serde` v1 and `serde-pickle` v1 Rust dependencies. `serde-pickle` is an arbitrary choice of a (de)serialization
  binary format, but it could be useful in the future having a way to inspect Rust structures from
  Python https://github.com/light-curve/light-curve-python/pull/145

### Changed

- Build environment: minimum support Rust version (MSRV) is bumped 1.57 -> 1.60
- Bump `light-curve-dmdt` 0.6.0 -> 0.7.1

### Fixed

- `BazinFit` and `VillarFit` docs are clarified for `.model()` and `ln_prior`
  usage https://github.com/light-curve/light-curve-python/issues/125 https://github.com/light-curve/light-curve-python/pull/146

## [0.6.3] 2022-12-23

No changes, it was accidentally released instead of `0.6.2`

## [0.6.2] 2022-12-27

- `OtsuSplit` implementation in
  Rust https://github.com/light-curve/light-curve-python/issues/120 https://github.com/light-curve/light-curve-python/pull/123

### Changed

- `light-curve-feature` 0.5.0 -> 0.5.2 https://github.com/light-curve/light-curve-python/pull/123
- `light-curve-dmdt` 0.5.0 -> 0.6.0
- `pyO3` 0.16.6 -> 0.17.3
- `rust-numpy` 0.16.2 -> 0.17.2
- CI: binary wheels are now built using our
  custom [manylinux/musllinux images](https://github.com/light-curve/base-docker-images), which include FFTW library
  optimised to use platform-specific SIMD instructions. It should give up to 50% performance gain for `Periodogram` at
  all Linux platforms but `x86_64` where we use MKL https://github.com/light-curve/light-curve-python/pull/134
- We don't provide binary wheels for Linux i686 anymore, please contact us if you need
  them https://github.com/light-curve/light-curve-python/pull/134
- wheel build dependency: `maturin` 0.13.x ->
  0.14.x https://github.com/light-curve/light-curve-python/issues/130 https://github.com/light-curve/light-curve-python/pull/135

### Deprecated

- cargo features "fftw-dynamic", "fftw-static" and "mkl" are renamed to "fftw-system", "fftw-source" and "fftw-mkl"
  correspondingly https://github.com/light-curve/light-curve-python/pull/137

### Fixed

— fix `threshold` method in Python according to Rust
implementation https://github.com/light-curve/light-curve-python/pull/123

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

- https://github.com/light-curve/light-curve-python/issues/110 Experimental warning message specifies a class
  name https://github.com/light-curve/light-curve-python/pull/111
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
- `ln_prior` argument for `BazinFit` and `VillarFit` constructors which can be one of: `None`, `str` literals (currently
  the only useful value is 'hosseinzadeh2020' for `VillarFit`) or `list[LnPrior1D]`
- `Cargo.lock` is used to build the release packages and it is added to sdist, all these should make builds more
  reproducible

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

- `check: bool = True` keyword argument for `__call__` and `many` methods of feature classes. It coulb be used to check
  if input arrays are valid

### Changed

- `gsl` is a default Cargo feature now, which means that GSL must be installed to build this package by standard Python
  tools like `pip install`
- `light-curve-feature` 0.3 -> 0.4.1 transition brings MCMC improvements, changing feature names of `BazinFit` and
  significant changes of `VillarFit` feature set

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
- Pure Python implementation of the most of the features are added, now Rust-implemented features live
  in `light_curve_ext` sub-package, while the Python implementation is in `light_curve_py`. Python-implemented feature
  extractors have an experimental status
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
