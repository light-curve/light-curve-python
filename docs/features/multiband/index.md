# Multiband features

Multiband features operate on light curves that have observations in **multiple photometric bands**.
Each observation carries a `band` label in addition to time, flux, and uncertainty.

These features are experimental — they live in the pure-Python part of the package and require
optional dependencies beyond the core install.

## Available features

| Feature | Description | Extra dependency |
|---------|-------------|------------------|
| `ColorOfMedian` | Difference of median magnitudes between two bands | — |
| `RainbowFit` | Multiband blackbody fit for photometric transients (e.g. supernovae) | `iminuit ≥ 2.21` |

## Quick start

```python
import numpy as np
from light_curve import ColorOfMedian, RainbowFit

# ColorOfMedian — no extra deps
feature = ColorOfMedian(blue_band="g", red_band="r")
color = feature(t, mag, sigma=sigma, band=band)

# RainbowFit — needs iminuit
# pip install light-curve[full]   or   pip install iminuit
band_wave_aa = {"g": 4770.0, "r": 6231.0, "i": 7625.0, "z": 9134.0}

# Standard blackbody (default)
rainbow = RainbowFit.from_angstrom(band_wave_aa, with_baseline=True)
params = rainbow(t, flux, sigma=flux_err, band=band)

# UV-extincted blackbody — adds log_intensity and lambda_scale fit parameters
rainbow_uv = RainbowFit.from_angstrom(band_wave_aa, with_baseline=True, spectral="blanketed")
params_uv = rainbow_uv(t, flux, sigma=flux_err, band=band)
```

### RainbowFit spectral models

| `spectral=` | Model | Extra parameters |
|-------------|-------|-----------------|
| `"planck"` *(default)* | Standard blackbody \(B_\nu(T)\) | — |
| `"blanketed"` | UV-extincted blackbody \(B_\nu(T)\cdot e^{-\tau}\), \(\tau = 10^{\texttt{log\_intensity}} \cdot e^{-\lambda/\texttt{lambda\_scale}}\) | `log_intensity`, `lambda_scale` (Å) |

The blanketed model is useful for supernovae with significant line blanketing in the UV.

See the [tutorial](tutorial.ipynb) for a worked example with synthetic data and visualisation,
and the [API reference](api.md) for full parameter documentation.
