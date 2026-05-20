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
rainbow = RainbowFit.from_angstrom(band_wave_aa, with_baseline=True)
params = rainbow(t, flux, sigma=flux_err, band=band)
```

See the [tutorial](tutorial.ipynb) for a worked example with synthetic data and visualisation,
and the [API reference](api.md) for full parameter documentation.
