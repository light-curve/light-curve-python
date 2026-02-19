# `light-curve` processing toolbox for Python

A Python wrapper for the [`light-curve-feature`](https://github.com/light-curve/light-curve-feature) and
[`light-curve-dmdt`](https://github.com/light-curve/light-curve-dmdt) Rust crates, providing high-performance
time-series feature extraction for astrophysics.

[![PyPI version](https://badge.fury.io/py/light-curve.svg)](https://pypi.org/project/light-curve/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/light-curve-python.svg)](https://anaconda.org/conda-forge/light-curve-python)
[![Conda Recipe](https://img.shields.io/badge/recipe-light--curve--python-green.svg)](https://anaconda.org/conda-forge/light-curve-python)
[![testing](https://github.com/light-curve/light-curve-python/actions/workflows/test.yml/badge.svg)](https://github.com/light-curve/light-curve-python/actions/workflows/test.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/light-curve/light-curve-python/main.svg)](https://results.pre-commit.ci/latest/github/light-curve/light-curve-python/main)

## Quick start

```sh
pip install 'light-curve[full]'
```

<!-- name: test_feature_evaluators_basic -->

```python
import light_curve as lc
import numpy as np

rng = np.random.default_rng(0)
n = 100  # observations per light curve

# Observation times in days (unevenly sampled, must be sorted)
t = np.sort(rng.uniform(0, 100, n))
# Magnitudes with a slight linear fade and measurement noise
m = 15.0 + 0.01 * t + rng.normal(0, 0.1, n)
err = np.full(n, 0.1)

# Combine features into a single extractor evaluated in one pass
extractor = lc.Extractor(lc.Amplitude(), lc.BeyondNStd(nstd=1), lc.LinearFit())

result = extractor(t, m, err)
print('\n'.join(f"{name} = {value:.4f}" for name, value in zip(extractor.names, result)))

# Extract a feature from 1000 light curves in parallel
light_curves = [
    (np.sort(rng.uniform(0, 100, n)), 15.0 + rng.normal(0, 0.2, n), np.full(n, 0.1))
    for _ in range(1000)
]
amplitudes = lc.Amplitude().many(light_curves)
print(f"Amplitude: mean = {np.mean(amplitudes):.3f} mag, std = {np.std(amplitudes):.3f} mag")
```

For conda, alternative package names, and platform-specific installation notes, see the [Installation](#installation)
section.

## Feature extractors

Most classes implement feature extractors useful for astrophysical source classification and characterization based
on light curves.

To list all available feature extractors:

<!-- name: test_feature_evaluators_list -->

```python
import light_curve as lc

print([x for x in dir(lc) if hasattr(getattr(lc, x), "names")])
```

To read the documentation for a specific extractor:

<!-- name: test_feature_evaluators_help -->

```python
import light_curve as lc

help(lc.BazinFit)
```

### Available features

See the complete list of available feature extractors and their documentation in the
[`light-curve-feature` Rust crate docs](https://docs.rs/light-curve-feature/latest/light_curve_feature/features/index.html).
Italic names are experimental features.
While we usually say "magnitude" and use "m" as the time-series value, some features are designed for
flux light curves.
The last column indicates whether a feature should be used with flux light curves only, magnitude light
curves only, or either.

<table>
  <tr>
    <th>Feature name</th>
    <th>Description</th>
    <th>Min data points</th>
    <th>Features number</th>
    <th>Flux/magnitude</th>
    <th>Default transform</th>
  </tr>
  <tr>
    <td>Amplitude</td>
    <td>Half amplitude of magnitude: <p align="center">$\displaystyle \frac{\max (m)-\min (m)}{2}$</p></td>
    <td align="center">1</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
    <td>identity</td>
  </tr>

  <tr>
    <td>AndersonDarlingNormal</td>
    <td>Unbiased Anderson–Darling normality test statistic:
<p align="left">$\displaystyle \left( 1+\frac{4}{N} -\frac{25}{N^{2}}\right) \times$<p>
<p align="right">$\times \left( -N-\frac{1}{N}\sum\limits_{i=0}^{N-1} (2i+1)\ln \Phi _{i} +(2(N-i)-1)\ln (1-\Phi _{i} )\right) ,$<p> where $\Phi _{i\ } \equiv \Phi (( m_{i} \ -\ \langle m\rangle ) /\sigma _{m})$ is the commutative distribution function of the standard normal distribution, $N-$ the number of observations, $\langle m\rangle -$ mean magnitude and $\sigma _{m} =\sqrt{\sum\limits_{i=0}^{N-1}( m_{i} -\langle m\rangle )^{2} /( N-1) \ }$ is the magnitude standard deviation</td>
    <td align="center">4</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
    <td>lg</td>
  </tr>

  <tr>
    <td>BazinFit</td>
    <td>Five fit parameters and goodness of fit (reduced $\chi ^{2}$) of the Bazin function developed for core-collapsed supernovae:
<p align="center">$\displaystyle f(t)=A\frac{\mathrm{e}^{-(t-t_{0} )/\tau _{fall}}}{1+\mathrm{e}^{-(t-t_{0} )/\tau _{rise}}} +B,$</p> where $f(t)-$ flux observation</td>
    <td align="center">6</td>
    <td align="center">1</td>
    <td>Flux only</td>
    <td>$A$ → mag, $B/A$, $t_0$ dropped, $\chi^2$ → $\ln(1+\chi^2)$</td>
  </tr>

  <tr>
    <td>BeyondNStd</td>
    <td>Fraction of observations beyond $n\sigma _{m}$ from the mean magnitude $\langle m\rangle $:
<p align="center">$\displaystyle \frac{\sum _{i} I_{|m-\langle m\rangle | >n\sigma _{m}} (m_{i} )}{N},$</p> where $I-$ an indicator function</td>
    <td align="center">2</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
    <td>identity</td>
  </tr>

  <tr>
    <td><i>ColorOfMedian</i> <br>(experimental)</td>
    <td>Magnitude difference between medians of two bands</td>
    <td align="center">2</td>
    <td align="center">1</td>
    <td>Magn only</td>
    <td>identity</td>
  </tr>

   <tr>
    <td>Cusum</td>
    <td>A range of cumulative sums:
<p align="center">$\displaystyle \max(S) - \min(S),$</p>
    where
<p align="center">$\displaystyle S_{j} \equiv \frac{1}{N\sigma_{m}} \sum_{i=0}^{j} (m_{i} - \langle m\rangle), \quad j \in \lbrace 1..N-1 \rbrace\;$</p></td>
    <td align="center">2</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
    <td>identity</td>
  </tr>

  <tr>
    <td>Eta</td>
    <td>Von Neummann $\eta $:
<p align="center">$\displaystyle \eta \equiv \frac{1}{(N-1)\sigma _{m}^{2}}\sum\limits _{i=0}^{N-2} (m_{i+1} -m_{i} )^{2}$</p></td>
    <td align="center">2</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
    <td>identity</td>
  </tr>

  <tr>
    <td>EtaE</td>
    <td>Modernisation of <b>Eta</b> for unevenly spaced time series:
<p align="center">$\displaystyle \eta ^{e} \equiv \frac{(t_{N-1} -t_{0} )^{2}}{(N-1)^{3}}\frac{\sum\limits_{i=0}^{N-2}\left(\frac{m_{i+1} -m_{i}}{t_{i+1} -t_{i}}\right)^{2}}{\sigma _{m}^{2}}$</p></td>
    <td align="center">2</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
    <td>lg</td>
  </tr>

  <tr>
    <td>ExcessVariance</td>
    <td>Measure of the variability amplitude:
<p align="center">$\displaystyle \frac{\sigma _{m}^{2} -\langle \delta ^{2} \rangle }{\langle m\rangle ^{2}},$</p> where $\langle \delta ^{2} \rangle -$ mean squared error</td>
    <td align="center">2</td>
    <td align="center">1</td>
    <td>Flux only</td>
    <td>identity</td>
  </tr>

  <tr>
    <td><i>FluxNNotDetBeforeFd</i><br>(experimental)</td>
    <td>Number of non-detections before the first detection</td>
    <td align="center">2</td>
    <td align="center">1</td>
    <td>Flux only</td>
    <td>identity</td>
  </tr>

  <tr>
    <td>InterPercentileRange</td>
    <td><p align="center">$\displaystyle Q(1-p)-Q(p),$</p> where $Q(n)$ and $Q(d)-$ $n$-th and $d$-th quantile of magnitude sample</td>
    <td align="center">1</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
    <td>identity</td>
  </tr>

  <tr>
    <td>Kurtosis</td>
    <td>Excess kurtosis of magnitude:
<p align="center">$\displaystyle \frac{N(N+1)}{(N-1)(N-2)(N-3)}\frac{\sum _{i} (m_{i} -\langle m\rangle )^{4}}{\sigma _{m}^{2}} -3\frac{(N+1)^{2}}{(N-2)(N-3)}$</p></td>
    <td align="center">4</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
    <td>arcsinh</td>
  </tr>

  <tr>
    <td>LinearFit</td>
    <td>The slope, its error and reduced $\chi ^{2}$ of the light curve in the linear fit of a magnitude light curve with respect to the observation error $\{\delta _{i}\}$:
<p align="center">$\displaystyle m_{i} \ =\ c\ +\ \text{slope} \ t_{i} \ +\ \delta _{i} \varepsilon _{i} ,$</p> where $c$ is a constant, $\{\varepsilon _{i}\}$ are standard distributed random variables</td>
    <td align="center">3</td>
    <td align="center">3</td>
    <td>Magn only</td>
    <td>identity</td>
  </tr>

  <tr>
    <td>LinearTrend</td>
    <td>The slope and its error of the light curve in the linear fit of a magnitude light curve without respect to the observation error $\{\delta _{i}\}$:
<p align="center">$\displaystyle m_{i} \ =\ c\ +\ \text{slope} \ t_{i} \ +\ \Sigma \varepsilon _{i} ,$</p> where $c$ and $\Sigma$ are constants, $\{\varepsilon _{i}\}$ are standard distributed random variables.</td>
    <td align="center">2</td>
    <td align="center">2</td>
    <td>Magn only</td>
    <td>identity</td>
  </tr>

  <tr>
    <td><i>MagnitudeNNotDetBeforeFd</i><br>(experimental)</td>
    <td>Number of non-detections before the first detection</td>
    <td align="center">2</td>
    <td align="center">1</td>
    <td>Magn only</td>
    <td>identity</td>
  </tr>

  <tr>
    <td>MagnitudePercentageRatio</td>
    <td>Magnitude percentage ratio:
<p align="center">$\displaystyle \frac{Q(1-n)-Q(n)}{Q(1-d)-Q(d)}$</p></td>
    <td align="center">1</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
    <td>identity</td>
  </tr>

  <tr>
    <td>MaximumSlope</td>
    <td>Maximum slope between two consecutive observations:
<p align="center">$\displaystyle \max_{i=0\dotsc N-2}\left| \frac{m_{i+1} -m_{i}}{t_{i+1} -t_{i}}\right|$</p></td>
    <td align="center">2</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
    <td>clipped lg</td>
  </tr>

  <tr>
    <td>Mean</td>
    <td>Mean magnitude:
<p align="center">$\displaystyle \langle m\rangle =\frac{1}{N}\sum\limits _{i} m_{i}$</p></td>
    <td align="center">1</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
    <td>identity</td>
  </tr>

  <tr>
    <td>MeanVariance</td>
    <td>Standard deviation to mean ratio:
<p align="center">$\displaystyle \frac{\sigma _{m}}{\langle m\rangle }$</p></td>
    <td align="center">2</td>
    <td align="center">1</td>
    <td>Flux only</td>
    <td>identity</td>
  </tr>

  <tr>
    <td>Median</td>
    <td>Median magnitude
    <td align="center">1</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
    <td>identity</td>
  </tr>

  <tr>
    <td>MedianAbsoluteDeviation</td>
    <td>Median of the absolute value of the difference between magnitude and its median:
<p align="center">$\displaystyle \mathrm{Median} (|m_{i} -\mathrm{Median} (m)|)$</p></td>
    <td align="center">1</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
    <td>identity</td>
  </tr>

  <tr>
    <td>MedianBufferRangePercentage</td>
    <td>Fraction of points within <p align="center">$\displaystyle \mathrm{Median} (m)\pm q\times (\max (m)-\min (m))/2$</p></td>
    <td align="center">1</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
    <td>identity</td>
  </tr>

  <tr>
    <td>OtsuSplit</td>
    <td>Difference of subset means, standard deviation of the lower subset, standard deviation of the upper
subset and lower-to-all observation count ratio for two subsets of magnitudes obtained by Otsu's method split.
<br>
<br>
Otsu's method is used to perform automatic thresholding. The algorithm returns a single threshold that separates values into two classes. This threshold is determined by minimizing intra-class intensity variance $\sigma^2_{W}=w_0\sigma^2_0+w_1\sigma^2_1$, or equivalently, by maximizing inter-class variance $\sigma^2_{B}=w_0 w_1 (\mu_1-\mu_0)^2$. When multiple extrema exist, the algorithm returns the minimum threshold.
   </td>
    <td align="center">2</td>
    <td align="center">4</td>
    <td>Flux or magn</td>
    <td>not supported</td>
  </tr>

  <tr>
    <td>PercentAmplitude</td>
    <td>Maximum deviation of magnitude from its median:
<p align="center">$\displaystyle \max_{i} |m_{i} \ -\ \text{Median}( m) |$</p></td>
    <td align="center">1</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
    <td>identity</td>
  </tr>

  <tr>
    <td>PercentDifferenceMagnitudePercentile</td>
    <td>Ratio of $p$-th inter-percentile range to the median:
<p align="center">$\displaystyle \frac{Q( 1-p) -Q( p)}{\text{Median}( m)}$</p></td>
    <td align="center">1</td>
    <td align="center">1</td>
    <td>Flux only</td>
    <td>clipped lg</td>
  </tr>

  <tr>
    <td>LinexpFit</td>
    <td>Four fit parameters and goodness of fit (reduced $\chi^{2}$) of the Linexp function developed for core-collapsed supernovae:
<p align="center">$\displaystyle f(t)=A\frac{t-t_{0}}{\tau}\exp\!\left(\frac{t-t_{0}}{\tau}\right)+B$</p></td>
    <td align="center">5</td>
    <td align="center">5</td>
    <td>Flux only</td>
    <td>$A$ → mag, $B/A$, $t_0$ dropped, $\chi^2$ → $\ln(1+\chi^2)$</td>
  </tr>

  <tr>
    <td><i>RainbowFit</i><br>(experimental)</td>
    <td>Seven fit parameters and goodness of fit (reduced $\chi ^{2}$). The Rainbow method is developed and detailed in <a href="https://arxiv.org/abs/2310.02916">Russeil+23</a>. This implementation is suited for transient objects. Bolometric flux and temperature functions are customizable; by default, Bazin and logistic functions are used:
<p align="center">$\displaystyle F_{\nu}(t, \nu) = \frac{\pi\,B\left(T(t),\nu\right)}{\sigma_\mathrm{SB}\,T(t)^{4}} \times F_\mathrm{bol}(t),$</p> where $F_{\nu}(t, \nu)-$ flux observation at a given wavelength</td>
    <td align="center">6</td>
    <td align="center">1</td>
    <td>Flux only</td>
    <td>identity</td>
  </tr>

  <tr>
    <td>ReducedChi2</td>
    <td>Reduced $\chi ^{2}$ of magnitude measurements:
<p align="center">$\displaystyle \frac{1}{N-1}\sum _{i}\left(\frac{m_{i} -\overline{m}}{\delta _{i}}\right)^{2} ,$</p> where $\overline{m} -$ weighted mean magnitude</td>
    <td align="center">2</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
    <td>$\ln(1+x)$</td>
  </tr>

  <tr>
    <td><i>Roms</i><br>(experimental)</td>
    <td>Robust median statistic: <p align="center">$\displaystyle \frac1{N-1} \sum_{i=0}^{N-1} \frac{|m_i - \mathrm{median}(m_i)|}{\sigma_i}$</p></td>
    <td align="center">2</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
    <td>identity</td>
  </tr>

  <tr>
    <td>Skew</td>
    <td>Skewness of magnitude:
<p align="center">$\displaystyle \frac{N}{(N-1)(N-2)}\frac{\sum _{i} (m_{i} -\langle m\rangle )^{3}}{\sigma _{m}^{3}}$</p></td>
    <td align="center">3</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
    <td>arcsinh</td>
  </tr>

  <tr>
    <td>StandardDeviation</td>
    <td>Standard deviation of magnitude:
<p align="center">$\displaystyle \sigma _{m} \equiv \sqrt{\sum _{i} (m_{i} -\langle m\rangle )^{2} /(N-1)}$</p></td>
    <td align="center">2</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
    <td>identity</td>
  </tr>

  <tr>
    <td>StetsonK</td>
    <td><b>Stetson K</b> coefficient describing light curve shape:
<p align="center">$\displaystyle \frac{\sum _{i}\left| \frac{m_{i} -\overline{m} }{\delta _{i}}\right| }{\sqrt{N\ \chi ^{2}}}$</p></td>
    <td align="center">2</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
    <td>identity</td>
  </tr>

  <tr>
    <td>VillarFit</td>
    <td>Seven fit parameters and goodness of fit (reduced $\chi ^{2}$) of the Villar function developed for supernovae classification:
<p align="center">$f(t)=c+\frac{A}{1+\exp\frac{-(t-t_{0} )}{\tau _{rise}}} \times f_{fall}(t),$</p>
<p align="center">$f_{fall}(t) = 1-\frac{\nu (t-t_{0} )}{\gamma }, ~~~ t< t_{0} +\gamma,$</p>
<p align="center">$f_{fall}(t) = (1-\nu )\exp\frac{-(t-t_{0} -\gamma )}{\tau _{fall}}, ~~~ t \geq t_{0} + \gamma.$</p>
where $f(t) -$ flux observation, $A, \gamma , \tau _{rise} , \tau _{fall}  >0$, $\nu \in [0;1)$</p>Here we introduce a new dimensionless parameter $\nu$ instead of the plateau slope $\beta$ from the original paper: $\nu \equiv -\beta \gamma /A$</td>
    <td align="center">8</td>
    <td align="center">8</td>
    <td>Flux only</td>
    <td>$A$ → mag, $c/A$, $t_0$ dropped, $\chi^2$ → $\ln(1+\chi^2)$</td>
  </tr>

  <tr>
    <td>WeightedMean</td>
    <td>Weighted mean magnitude:
<p align="center">$\displaystyle \overline{m} \equiv \frac{\sum _{i} m_{i} /\delta _{i}^{2}}{\sum _{i} 1/\delta _{i}^{2}}$</p></td>
    <td align="center">1</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
    <td>identity</td>
  </tr>

</table>

### Meta-features

Meta-features accept other feature extractors and apply them to pre-processed data.

#### Periodogram

This feature transforms time-series data into the Lomb-Scargle periodogram, providing an estimate of the power
spectrum. The `peaks` argument specifies how many of the most significant spectral density peaks to return. For
each peak, its period and signal-to-noise ratio are returned:

$$
\text{signal to noise of peak} \equiv \frac{P(\omega_\mathrm{peak}) - \langle P(\omega) \rangle}{\sigma\_{P(\omega)}}
$$

The optional `features` argument accepts a list of additional feature extractors, which are applied to the power
spectrum: frequency is passed as "time," power spectrum is passed as "magnitude," and no uncertainties are set.

#### Bins

Bins the time series into windows of width $\mathrm{window}$ with respect to some $\mathrm{offset}$.
The $j$-th bin spans $[j \cdot \mathrm{window} + \mathrm{offset},\ (j + 1) \cdot \mathrm{window} + \mathrm{offset}]$.

The binned time series is defined by
$$t_j^* = (j + \frac12) \cdot \mathrm{window} + \mathrm{offset},$$
$$m_j^* = \frac{\sum{m_i / \delta_i^2}}{\sum{\delta_i^{-2}}},$$
$$\delta_j^* = \frac{N_j}{\sum{\delta_i^{-2}}},$$
where $N_j$ is the number of observations in the bin and all sums are over observations within that bin.

### Parametric fit features

`BazinFit`, `LinexpFit`, and `VillarFit` fit parametric functions to transient flux light curves (see
[Available features](#available-features) for the function definitions). Unlike other extractors,
they require an explicit `algorithm` argument selecting the optimization method.

The available algorithms depend on the compile-time Cargo features (see
[Build from source](#build-from-source)):

| Algorithm | Requires | Description |
|-----------|----------|-------------|
| `"mcmc"` | always available | MCMC ensemble sampler; robust but slow |
| `"ceres"` | `ceres-source` or `ceres-system` | Ceres trust-region solver; fast, gradient-based |
| `"mcmc-ceres"` | `ceres-source` or `ceres-system` | MCMC exploration followed by Ceres refinement |
| `"lmsder"` | `gsl` | Levenberg-Marquardt via GSL |
| `"mcmc-lmsder"` | `gsl` | MCMC exploration followed by LMSDER refinement |

The hybrid `"mcmc-ceres"` and `"mcmc-lmsder"` algorithms run MCMC first to broadly explore the
parameter space, then hand off to the gradient-based solver for precise convergence. This typically
gives the best balance of robustness and final accuracy.

By default, initial parameter values and bounds are estimated from the data. Override them with the
`init` and `bounds` arguments (supported by MCMC-based algorithms only), each a list of values or
`None`s to keep data-derived defaults for individual parameters. `BazinFit` has 5 parameters, `LinexpFit` has 4, `VillarFit` has 7; run `help()` on the
respective class for the parameter order.

The `ln_prior` argument sets the MCMC prior. It accepts `None` / `"no"` (flat prior, the default),
a named string literal, or a list of `light_curve.ln_prior.LnPrior1D` objects for per-parameter
distributions (see `help(light_curve.ln_prior)` for the available distribution types).
`VillarFit` additionally supports `"hosseinzadeh2020"`, a prior from Hosseinzadeh et al. 2020 that
encodes supernova physics constraints; it assumes time values are in days.

Pass `transform=True` to convert the raw fit parameters to a magnitude-like representation: the
amplitude becomes a magnitude, the baseline is normalized by the amplitude, the reference time is
dropped, and the reduced chi^2 is log-scaled. Run `help(lc.BazinFit)` for the exact definition.

For the experimental multi-band analogue, see [Rainbow Fit](#rainbow-fit) below.

### Multi-band features

As of v0.8, experimental extractors support multi-band light-curve inputs.

<!-- name: test_multiband_experimental_features -->

```python
import numpy as np
from light_curve.light_curve_py import LinearFit

t = np.arange(20, dtype=float)
m = np.arange(20, dtype=float)
sigma = np.full_like(t, 0.1)
bands = np.array(["g"] * 10 + ["r"] * 10)
feature = LinearFit(bands=["g", "r"])
values = feature(t, m, sigma, bands)
print(values)
```

#### Rainbow Fit

Rainbow ([Russeil+23](https://arxiv.org/abs/2310.02916)) is a black-body parametric model for transient light
curves. By default, it uses the Bazin function as a model for bolometric flux evolution and a logistic function
for temperature evolution. The user may customize the model by providing their own functions for bolometric flux
and temperature evolution. This example demonstrates the reconstruction of a synthetic light curve with this model.
`RainbowFit` requires the `iminuit` package.

<!-- name: test_rainbow_fit_example -->

```python
import numpy as np
from light_curve.light_curve_py import RainbowFit


def bb_nu(wave_aa, T):
    """Black-body spectral model"""
    nu = 3e10 / (wave_aa * 1e-8)
    return 2 * 6.626e-27 * nu ** 3 / 3e10 ** 2 / np.expm1(6.626e-27 * nu / (1.38e-16 * T))


# Effective wavelengths in Angstrom
band_wave_aa = {"g": 4770.0, "r": 6231.0, "i": 7625.0, "z": 9134.0}

# Parameter values
reference_time = 60000.0  # time close to the peak time
# Bolometric flux model parameters
amplitude = 1.0  # bolometric flux semiamplitude, arbitrary (non-spectral) flux/luminosity units
rise_time = 5.0  # exponential growth timescale, days
fall_time = 30.0  # exponential decay timescale, days
# Temperature model parameters
Tmin = 5e3  # temperature on +infinite time, kelvins
delta_T = 10e3  # (Tmin + delta_T) is temperature on -infinite time, kelvins
k_sig = 4.0  # temperature evolution timescale, days

rng = np.random.default_rng(0)
t = np.sort(rng.uniform(reference_time - 3 * rise_time, reference_time + 3 * fall_time, 1000))
band = rng.choice(list(band_wave_aa), size=len(t))
waves = np.array([band_wave_aa[b] for b in band])

# Temperature evolution is a sigmoid function
temp = Tmin + delta_T / (1.0 + np.exp((t - reference_time) / k_sig))
# Bolometric flux evolution is the Bazin function
lum = amplitude * np.exp(-(t - reference_time) / fall_time) / (
        1.0 + np.exp(-(t - reference_time) / rise_time))

# Spectral flux density for each given pair of time and passband
flux = np.pi * bb_nu(waves, temp) / (5.67e-5 * temp ** 4) * lum
# S/N = 5 for minimum flux, scale for Poisson noise
flux_err = np.sqrt(flux * np.min(flux) / 5.0)
flux += rng.normal(0.0, flux_err)

feature = RainbowFit.from_angstrom(band_wave_aa, with_baseline=False)
values = feature(t, flux, sigma=flux_err, band=band)
print(dict(zip(feature.names, values)))
print(f"Goodness of fit: {values[-1]}")
```

Note that while the data generation above uses approximate physical constant values, `RainbowFit` uses CODATA 2018
values internally.

### Experimental extractors

The package consists of two parts: a wrapper for the
[`light-curve-feature` Rust crate](https://crates.io/crates/light-curve-feature) (`light_curve_ext` sub-package)
and a pure-Python sub-package `light_curve_py`.
We use the Python implementation to test the Rust implementation and to develop new experimental extractors.
The Python implementation is significantly slower for most extractors and does not provide the same full
feature set as the Rust implementation, but it does include some extractors you may find useful.

You can use extractors from either implementation directly:

<!-- name: test_experimental_extractors -->

```python
import numpy as np
from numpy.testing import assert_allclose
from light_curve.light_curve_ext import LinearTrend as RustLinearTrend
from light_curve.light_curve_py import LinearTrend as PythonLinearTrend

rust_fe = RustLinearTrend()
py_fe = PythonLinearTrend()

n = 100
t = np.sort(np.random.normal(size=n))
m = 3.14 * t - 2.16 + np.random.normal(size=n)

assert_allclose(rust_fe(t, m), py_fe(t, m),
                err_msg="Python and Rust implementations must provide the same result")
```

This should print a warning about the experimental status of the Python class.

## Performance

The package is designed for high throughput. The following techniques help extract features with minimal overhead.

### `sorted` and `check` parameters

The `sorted=True` argument tells the extractor that `t` is already sorted in ascending order, and `check=False`
disables validation of NaN/inf values. The defaults are `sorted=None` (the array will be validated and an error raised if unsorted) and
`check=True`. Passing invalid inputs without validation can cause incorrect results or crashes.

### Batch processing with `.many()`

Each extractor has a `.many()` method that processes a collection of light curves in parallel across all
available CPU cores by default. It accepts either a list of `(t, m[, sigma])` tuples or an Arrow array
(see below).

```python
import light_curve as lc
import numpy as np

rng = np.random.default_rng(0)
# A list of (t, m, sigma) tuples, one per light curve
light_curves = [
    (np.sort(rng.random(50)), rng.random(50), rng.random(50) * 0.1)
    for _ in range(1000)
]

results = lc.Amplitude().many(light_curves)
```

### Arrow input

The `.many()` method also accepts Arrow arrays instead of a list of tuples, enabling zero-copy data access
from Arrow-compatible libraries. The input must be a `List<Struct<t, m[, sigma]>>` array where all fields
share the same float dtype. Any library implementing the
[Arrow C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html) is supported.
All light curves are processed in parallel using Rust threads, bypassing the GIL entirely.

#### nested-pandas

[nested-pandas](https://nested-pandas.readthedocs.io) extends pandas with nested column support,
useful for working with catalog data like ZTF or Rubin LSST.
Install with `pip install nested-pandas s3fs universal-pathlib`.

<!-- name: test_many_nested_pandas -->

```python
import light_curve as lc
import nested_pandas as npd
import numpy as np
import pyarrow as pa
from upath import UPath

# Read a ZTF DR23 HiPSCat partition with nested light curves
s3_path = UPath(
    "s3://ipac-irsa-ztf/contributed/dr23/lc/hats/ztf_dr23_lc-hats/dataset/Norder=6/Dir=30000/Npix=34623/part0.snappy.parquet",
    anon=True,
)
ndf = npd.read_parquet(
    s3_path,
    columns=["objectid", "lightcurve.hmjd", "lightcurve.mag", "lightcurve.magerr"],
)

# Select rows with more than 10 observations
ndf = ndf.loc[ndf["lightcurve"].list_lengths > 10]

# We must have subcolumns to be the same dtype and be in t, m, sigma order, so we rename and convert time.
# Create a float32 time column and drop the original double-precision one
ndf["lightcurve.t"] = np.asarray(ndf["lightcurve.hmjd"] - 58000, dtype=np.float32)
ndf["lightcurve"] = ndf[["lightcurve.t", "lightcurve.mag", "lightcurve.magerr"]]

# Extract features directly from the Arrow-backed nested column
feature = lc.Extractor(lc.Amplitude(), lc.EtaE(), lc.InterPercentileRange(quantile=0.25))
result = feature.many(pa.array(ndf["lightcurve"]), n_jobs=-1)

# Assign features back to the dataframe
ndf = ndf.assign(**dict(zip(feature.names, result.T)))
print(ndf.head())
```

#### PyArrow

[PyArrow](https://arrow.apache.org/docs/python/) is the reference Python implementation of Apache Arrow.
Install with `pip install pyarrow`.

<!-- name: test_many_pyarrow -->

```python
import light_curve as lc
import numpy as np
import pyarrow as pa

rng = np.random.default_rng(42)
n_lc = 10

# Build a PyArrow List<Struct<t, m, sigma>> array
struct_type = pa.struct([("t", pa.float64()), ("m", pa.float64()), ("sigma", pa.float64())])
lcs_arrow = pa.array(
    [
        [{"t": t, "m": m, "sigma": s} for t, m, s in zip(*[np.sort(rng.random(50)) for _ in range(3)])]
        for _ in range(n_lc)
    ],
    type=pa.list_(struct_type),
)

feature = lc.Extractor(lc.Kurtosis(), lc.Skew(), lc.ReducedChi2())
result = feature.many(lcs_arrow, sorted=True, check=False, n_jobs=4)
print(f"Features: {feature.names}")
print(f"Results shape: {result.shape}")
```

#### Polars

[Polars](https://docs.pola.rs) is a fast DataFrame library built on Arrow.
Install with `pip install polars`.

<!-- name: test_many_polars -->

```python
import light_curve as lc
import numpy as np
import polars as pl

rng = np.random.default_rng(42)

# Start with a flat DataFrame, as you might get from a database or CSV
object_id = np.repeat(np.arange(10), 50)
t = np.sort(rng.random(500))
m = rng.random(500)
sigma = rng.random(500)

df = pl.DataFrame({"object_id": object_id, "t": t, "m": m, "sigma": sigma})

# Group by object_id into List<Struct<t, m, sigma>>
nested = df.group_by("object_id").agg(pl.struct("t", "m", "sigma").alias("lc"))

feature = lc.Extractor(lc.Amplitude(), lc.BeyondNStd(nstd=2), lc.LinearFit(), lc.StetsonK())
result = feature.many(nested["lc"], sorted=True, check=False, n_jobs=1)

# Join feature columns back to the nested DataFrame
nested = nested.with_columns(
    [pl.Series(name, result[:, i]) for i, name in enumerate(feature.names)]
)
print(nested)
```

### Benchmarks

Run all benchmarks from the Python project folder with
`python3 -mpytest --benchmark-enable tests/test_w_bench.py`, or with slow benchmarks disabled:
`python3 -mpytest -m "not (nobs or multi)" --benchmark-enable tests/test_w_bench.py`.

Below we benchmark the Rust implementation (`rust`) against the [`feets`](https://feets.readthedocs.io/en/latest/)
package and our own Python implementation (`lc_py`) for a light curve with n=1000 observations.

![Benchmarks, Rust is much faster](https://github.com/light-curve/light-curve-python/raw/main/light-curve/.readme/benchplot_v2.png)

The Rust implementation outperforms the alternatives by a factor of 1.5–50.
This allows extracting a large set of "cheap" features in well under one millisecond for n=1000.
The performance of parametric fits (`BazinFit` and `VillarFit`) and `Periodogram` depends on their parameters,
but the typical extraction time including these features is 20–50 ms for a few hundred observations.

![Benchmark for different number of observations](https://github.com/light-curve/light-curve-python/raw/main/light-curve/.readme/nobs_bench_v2.png)

Benchmark results for both the pure-Python and Rust implementations as a function of the number of observations.
Both axes are on a logarithmic scale.

![Benchmark for multithreading and multiprocessing](https://github.com/light-curve/light-curve-python/raw/main/light-curve/.readme/multi_bench_v2.png)

Processing time per light curve for the feature subset from the first benchmark, as a function of the number of
CPU cores used. The dataset consists of 10,000 light curves with 1,000 observations each.

See the benchmarks described in more detail in
["Performant feature extraction for photometric time series"](https://arxiv.org/abs/2302.10837).

## dm-dt map

In addition to the feature extractors above, the package provides a separate dm–dt mapping tool.
The dm–dt map is a 2-D histogram of magnitude differences (dm) versus time differences (dt) for all pairs
of observations in a light curve. It is commonly used as an input representation for machine learning
classifiers rather than as a scalar feature.

The `DmDt` class (a Python wrapper for the
[`light-curve-dmdt` Rust crate](https://crates.io/crates/light-curve-dmdt)) implements this mapper,
following [Mahabal et al. 2011](https://ui.adsabs.harvard.edu/abs/2011BASI...39..387M/abstract) and
[Soraisam et al. 2020](https://ui.adsabs.harvard.edu/abs/2020ApJ...892..112S/abstract).

<!-- name: test_dmdt -->

```python
import numpy as np
from light_curve import DmDt
from numpy.testing import assert_array_equal

dmdt = DmDt.from_borders(min_lgdt=0, max_lgdt=np.log10(3), max_abs_dm=3, lgdt_size=2, dm_size=4, norm=[])

t = np.array([0, 1, 2], dtype=np.float32)
m = np.array([0, 1, 2], dtype=np.float32)

desired = np.array(
    [
        [0, 0, 2, 0],
        [0, 0, 0, 1],
    ]
)
actual = dmdt.points(t, m)

assert_array_equal(actual, desired)
```

## Installation

```sh
pip install 'light-curve[full]'
# or
conda install -c conda-forge light-curve-python
```

The `full` extra installs all optional Python dependencies required by experimental features.
We also provide a `light-curve-python` package as an alias for `light-curve[full]`.

The minimum supported Python version is 3.10.
Binary CPython distributions are available via [PyPI](https://pypi.org/project/light-curve/)
and [Anaconda](https://anaconda.org/conda-forge/light-curve-python) for various platforms and architectures.
On PyPI, we publish wheels for the stable CPython ABI, ensuring compatibility with future CPython 3 versions.

### Support matrix

| Arch \ OS   | Linux glibc 2.17+ | Linux musl 1.2+                | macOS                 | Windows https://github.com/light-curve/light-curve-python/issues/186 |
|-------------|-------------------|--------------------------------|-----------------------|----------------------------------------------------------------------|
| **x86-64**  | PyPI (MKL), conda | PyPI (MKL)                     | PyPI macOS 15+, conda | PyPI, conda (both no Ceres, no GSL)                                  |
| **i686**    | src               | src                            | —                     | not tested                                                           |
| **aarch64** | PyPI              | PyPI                           | PyPI macOS 14+, conda | not tested                                                           |
| **ppc64le** | src               | not tested (no Rust toolchain) | —                     | —                                                                    |

- **PyPI / conda**: A binary wheel or package is available on pypi.org or anaconda.org.
  Local building is not required; the only prerequisite is a recent version of `pip` or `conda`.
  For Linux x86-64, PyPI wheels are built with Intel MKL for improved periodogram performance,
  which is not the default build option.
  For Windows x86-64, all distributions exclude Ceres and GSL support.
- **src**: The package has been confirmed to build and pass unit tests locally,
  but CI does not test or publish packages for this platform.
  See the [Build from source](#build-from-source) section for details.
  Please open an issue or pull request if you encounter build problems or would like us to distribute
  packages for these platforms.
- **not tested**: Building from source has not been tested.
  Please report build status via issue, PR, or email.

macOS wheels require relatively recent OS versions; please open an issue if you need support for older macOS
versions (see https://github.com/light-curve/light-curve-python/issues/376 for details).

We no longer publish PyPy wheels ([#345](https://github.com/light-curve/light-curve-python/issues/345))
or the PPC64le CPython glibc wheel ([#479](https://github.com/light-curve/light-curve-python/issues/479));
please open an issue if you need either.

Free-threaded Python
([experimental in Python 3.13](https://docs.python.org/3.13/whatsnew/3.13.html#free-threaded-cpython),
[officially supported in Python 3.14+](https://docs.python.org/3.14/whatsnew/3.14.html#whatsnew314-pep779))
is supported when built from source.
No pre-built distributions are currently provided; please comment on the relevant issues if you need them:
[PyPI binary wheel issue](https://github.com/light-curve/light-curve-python/issues/500),
[conda-forge package issue](https://github.com/conda-forge/light-curve-python-feedstock/issues/11).
Note that for expensive features, the GIL-enabled interpreter with `.many()` achieves performance on par with
the free-threaded interpreter using Python threads. For inexpensive extractors, `.many()` still reduces
Rust–Python interaction overhead significantly.

## Developer guide

### Prepare environment

Install a recent Rust toolchain and Python 3.10 or higher.
It is recommended to use [`rustup`](https://rustup.rs/) to install the Rust toolchain and keep it updated
with `rustup update`.

Clone the repository, then create and activate a virtual environment:

```bash
git clone https://github.com/light-curve/light-curve-python.git
cd light-curve-python/light-curve
python3 -m venv venv
source venv/bin/activate
```

Install the package in editable mode (see [Build from source](#build-from-source) for more details):

```bash
python -mpip install maturin
# --release takes longer to build but produces a faster package
# Add other Cargo flags if needed, e.g. --no-default-features --features=ceres-source
maturin develop --extras=dev
```

On subsequent runs, activate the environment with `source venv/bin/activate` and rebuild Rust code with
`maturin develop`. Python-only changes require no rebuild. The `--extras=dev` flag is only needed once to
install development dependencies.

It is also recommended to install `pre-commit` hooks to keep the codebase clean.
Install via `pip` (see the [documentation](https://pre-commit.com/#install) for other options), then run:

```bash
pre-commit install
```

### Run tests and benchmarks

All test dependencies are installed with `--extras=dev`. Run the tests with:

```bash
python -mpytest
```

Benchmarks are disabled by default; enable them with `--benchmark-enable`:

```bash
python -mpytest --benchmark-enable
```

See the [Benchmarks](#benchmarks) section in [Performance](#performance) for more details.

### Build from source

#### Dependencies and Cargo features

The package has a number of compile-time Cargo features, mostly controlling which C/C++ dependencies are used.
Pass the desired features to `maturin` with `--features`; it is also recommended to use `--no-default-features`
to avoid building unnecessary dependencies.

Available features:

- `abi3` (default) — enables CPython ABI3 compatibility. Disable for other interpreters or if you have a
  specific reason (our benchmarks show no performance difference). ABI3 is not supported by free-threaded
  CPython or PyPy.
- `ceres-source` (default) — builds [Ceres solver](http://ceres-solver.org/) from source. Requires a C++
  compiler and CMake. Known to not work on Windows. Used as an optional optimization backend for `BazinFit`
  and `VillarFit`.
- `ceres-system` — links against a system-installed Ceres dynamic library instead of building from source.
- `mkl` — enables the [FFTW](http://www.fftw.org/) interface with the Intel MKL backend for the fast
  periodogram. Intel MKL is downloaded automatically during the build. Highly recommended for Intel CPUs.
  Without this feature, the pure-Rust [RustFFT](https://crates.io/crates/rustfft) backend is used.
- `gsl` (default) — enables [GNU Scientific Library](https://www.gnu.org/software/gsl/) support. Requires a
  compatible GSL version installed on your system. Used as an optional optimization backend for `BazinFit`
  and `VillarFit`.
- `mimalloc` (default) — enables the [mimalloc](https://github.com/microsoft/mimalloc) memory allocator.
  Benchmarks show up to a 2× speedup for some cheap features, though it may increase memory usage.

#### Build with maturin

[maturin](https://www.maturin.rs/) is the recommended tool for building Rust-backed Python packages.
This example builds with minimal system dependencies:

```bash
python -mpip install maturin
maturin build --release --locked --no-default-features --features=abi3,mimalloc
````

The `--release` flag enables release mode (slower build, faster execution), `--locked` ensures reproducible
builds, and `--no-default-features --features=abi3,mimalloc` selects only ABI3 compatibility and the mimalloc
allocator.

#### Build with `build`

You can also build the package with [build](https://build.pypa.io/):

```bash
python -mpip install build
MATURIN_PEP517_ARGS="--locked --no-default-features --features=abi3,mimalloc" python -m build
```

#### Build with cibuildwheel

[cibuildwheel](https://cibuildwheel.pypa.io/) builds wheels on CI servers; we use it with GitHub Actions.
You can also use it locally to build wheels for your platform (replace the identifier with one from
the [list of supported platforms](https://cibuildwheel.pypa.io/en/stable/options/#build-skip)):

```bash
python -mpip install cibuildwheel
python -m cibuildwheel --only=cp310-manylinux_x86_64
```

Note that different Cargo feature sets are used for different platforms, as defined in `pyproject.toml`.
Windows wheels must be built on Windows; Linux wheels can be built on any platform with Docker (Qemu may be
needed for cross-architecture builds); macOS wheels must be built on macOS.
macOS builds also require `MACOSX_DEPLOYMENT_TARGET` to match the current macOS version, since Homebrew
dependencies are built against it:

```bash
export MACOSX_DEPLOYMENT_TARGET=$(sw_vers -productVersion | awk -F '.' '{print $1"."0}')
python -m cibuildwheel --only=cp310-macosx_arm64
unset MACOSX_DEPLOYMENT_TARGET
```

Since we use ABI3 compatibility, a wheel built for CPython 3.10 works with any later CPython 3 version.

## Citation

If you found this project useful for your research, please
cite [Malanchev et al., 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.502.5147M/abstract):

```bibtex
@ARTICLE{2021MNRAS.502.5147M,
       author = {{Malanchev}, K.~L. and {Pruzhinskaya}, M.~V. and {Korolev}, V.~S. and {Aleo}, P.~D. and {Kornilov}, M.~V. and {Ishida}, E.~E.~O. and {Krushinsky}, V.~V. and {Mondon}, F. and {Sreejith}, S. and {Volnova}, A.~A. and {Belinski}, A.~A. and {Dodin}, A.~V. and {Tatarnikov}, A.~M. and {Zheltoukhov}, S.~G. and {(The SNAD Team)}},
        title = "{Anomaly detection in the Zwicky Transient Facility DR3}",
      journal = {\mnras},
     keywords = {methods: data analysis, astronomical data bases: miscellaneous, stars: variables: general, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Solar and Stellar Astrophysics},
         year = 2021,
        month = apr,
       volume = {502},
       number = {4},
        pages = {5147-5175},
          doi = {10.1093/mnras/stab316},
archivePrefix = {arXiv},
       eprint = {2012.01419},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021MNRAS.502.5147M},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
