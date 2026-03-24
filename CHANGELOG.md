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

--

### Security

--

## [0.12.0] 2026-03-24

### Added

- `BazinFit`, `LinexpFit`, and `VillarFit` gained `nuts`, `nuts-ceres`, and `nuts-lmsder` algorithms using NUTS (No-U-Turn Sampler) for gradient-based Hamiltonian Monte Carlo optimization. New `nuts_ntune` and `nuts_niter` parameters control the number of tuning and drawing iterations respectively https://github.com/light-curve/light-curve-python/pull/635

### Changed

- Bump `light-curve-feature` to 0.13.0, `rand` to 0.10, `rand_xoshiro` to 0.8 https://github.com/light-curve/light-curve-python/pull/648
- **Build breaking**: Minimum supported Rust version (MSRV) bumped from 1.85 to 1.88 https://github.com/light-curve/light-curve-python/pull/648

### Deprecated

--

### Removed

--

### Fixed

- `.many(arrow_array)` now checks for nulls, and raises an error if any nulls presented. Previously we used all the values, even masked, which may cause unexpected results https://github.com/light-curve/light-curve-python/pull/629

### Security

--

