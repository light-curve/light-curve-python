# `light-curve` processing toolbox for Python

The Python wrapper for Rust [`light-curve-feature`](https://github.com/light-curve/light-curve-feature)
and [`light-curve-dmdt`](https://github.com/light-curve/light-curve-dmdt) packages which gives a collection of
high-performant time-series feature extractors.

[![PyPI version](https://badge.fury.io/py/light-curve.svg)](https://pypi.org/project/light-curve/)
![testing](https://github.com/light-curve/light-curve-python/actions/workflows/test.yml/badge.svg)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/light-curve/light-curve-python/master.svg)](https://results.pre-commit.ci/latest/github/light-curve/light-curve-python/master)

## Installation

```sh
python3 -mpip install 'light-curve[full]'
```

`full` extras would install the package with all optional Python dependencies required by experimental features.
We also provide `light-curve-python` package which is just an "alias" to the main `light-curve[full]` package.

Minimum supported Python version is 3.8.
We provide binary CPython wheels via [PyPi](https://pypi.org/project/light-curve/) for a number of platforms and
architectures.
We also provide binary wheels for stable CPython ABI, so the package is guaranteed to work with all future CPython3
versions.

### Support matrix

| Arch \ OS   | Linux glibc 2.17+ | Linux musl 1.1+                | macOS     | Windows https://github.com/light-curve/light-curve-python/issues/186 |
|-------------|-------------------|--------------------------------|-----------|----------------------------------------------------------------------|
| **x86-64**  | wheel (MKL)       | wheel (MKL)                    | wheel 12+ | wheel (no Ceres, no GSL)                                             |
| **i686**    | src               | src                            | —         | not tested                                                           |
| **aarch64** | wheel             | wheel                          | wheel 14+ | not tested                                                           |
| **ppc64le** | wheel             | not tested (no Rust toolchain) | —         | —                                                                    |

- "wheel": binary wheel is available on pypi.org, local building is not required for the platform, the only
  pre-requirement is a recent `pip` version. For Linux x86-64 we provide binary wheels built with Intel MKL for better
  periodogram performance, which is not a default build option. For Windows x86-64 we provide wheel with no Ceres and no
  GSL support, which is not a default build option.
- "src": the package is confirmed to be built and pass unit tests locally, but testing and package building is not
  supported by CI. It is required to have the [GNU scientific library (GSL)](https://www.gnu.org/software/gsl/) v2.1+
  and the [Rust toolchain](https://rust-lang.org) v1.67+ to install it via `pip install`. `ceres-solver` and `fftw` may
  be installed locally or built from source, in the later case you would also need C/C++ compiler and `cmake`.
- "not tested": building from the source code is not tested, please report us building status via issue/PR/email.

macOS wheels require relatively new OS versions, please open an issue if you require support of older Macs,
see https://github.com/light-curve/light-curve-python/issues/376 for the details.

We stopped publishing PyPy wheels (https://github.com/light-curve/light-curve-python/issues/345), please feel free to
open an issue if you need them.

## Feature evaluators

Most of the classes implement various feature evaluators useful for light-curve based
astrophysical source classification and characterisation.

<!-- name: test_feature_evaluators_basic -->

```python
import light_curve as lc
import numpy as np

# Time values can be non-evenly separated but must be an ascending array
n = 101
t = np.linspace(0.0, 1.0, n)
perfect_m = 1e3 * t + 1e2
err = np.sqrt(perfect_m)
m = perfect_m + np.random.normal(0, err)

# Half-amplitude of magnitude
amplitude = lc.Amplitude()
# Fraction of points beyond standard deviations from mean
beyond_std = lc.BeyondNStd(nstd=1)
# Slope, its error and reduced chi^2 of linear fit
linear_fit = lc.LinearFit()
# Feature extractor, it will evaluate all features in more efficient way
extractor = lc.Extractor(amplitude, beyond_std, linear_fit)

# Array with all 5 extracted features
result = extractor(t, m, err, sorted=True, check=False)

print('\n'.join(f"{name} = {value:.2f}" for name, value in zip(extractor.names, result)))

# Run in parallel for multiple light curves:
results = amplitude.many(
    [(t[:i], m[:i], err[:i]) for i in range(n // 2, n)],
    n_jobs=-1,
    sorted=True,
    check=False,
)
print("Amplitude of amplitude is {:.2f}".format(np.ptp(results)))
```

If you're confident in your inputs you could use `sorted = True` (`t` is in ascending order)
and `check = False` (no NaNs in inputs, no infs in `t` or `m`) for better performance.
Note that if your inputs are not valid and are not validated by
`sorted=None` and `check=True` (default values) then all kind of bad things could happen.

Print feature classes list
<!-- name: test_feature_evaluators_list -->

```python
import light_curve as lc

print([x for x in dir(lc) if hasattr(getattr(lc, x), "names")])
```

Read feature docs
<!-- name: test_feature_evaluators_help -->

```python
import light_curve as lc

help(lc.BazinFit)
```

### Available features

See the complete list of available feature evaluators and documentation
in [`light-curve-feature` Rust crate docs](https://docs.rs/light-curve-feature/latest/light_curve_feature/features/index.html).
Italic names are experimental features.
While we usually say "magnitude" and use "m" as a time-series value, some of the features are supposed to be used with
flux light-curves.
The last column indicates whether the feature should be used with flux light curves only, magnitude light curves only,
or any kind of light curves.

<table>
  <tr>
    <th>Feature name</th>
    <th>Description</th>
    <th>Min data points</th>
    <th>Features number</th>
    <th>Flux/magnitude</th>
  </tr>
  <tr>
    <td>Amplitude</td>
    <td>Half amplitude of magnitude: <p align="center">$\displaystyle \frac{\max (m)-\min (m)}{2}$</p></td>
    <td align="center">1</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
  </tr>

  <tr>
    <td>AndersonDarlingNormal</td>
    <td>Unbiased Anderson–Darling normality test statistic:
<p align="left">$\displaystyle \left( 1+\frac{4}{N} -\frac{25}{N^{2}}\right) \times$<p>
<p align="right">$\times \left( -N-\frac{1}{N}\sum\limits_{i=0}^{N-1} (2i+1)\ln \Phi _{i} +(2(N-i)-1)\ln (1-\Phi _{i} )\right) ,$<p> where $\Phi _{i\ } \equiv \Phi (( m_{i} \ -\ \langle m\rangle ) /\sigma _{m})$ is the commutative distribution function of the standard normal distribution, $N-$ the number of observations, $\langle m\rangle -$ mean magnitude and $\sigma _{m} =\sqrt{\sum\limits_{i=0}^{N-1}( m_{i} -\langle m\rangle )^{2} /( N-1) \ }$ is the magnitude standard deviation</td>
    <td align="center">4</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
  </tr>

  <tr>
    <td>BazinFit</td>
    <td>Five fit parameters and goodness of fit (reduced $\chi ^{2}$ of the Bazin function developed for core-collapsed supernovae:
<p align="center">$\displaystyle f(t)=A\frac{\mathrm{e}^{-(t-t_{0} )/\tau _{fall}}}{1+\mathrm{e}^{-(t-t_{0} )/\tau _{rise}}} +B,$</p> where $f(t)-$ flux observation</td>
    <td align="center">6</td>
    <td align="center">1</td>
    <td>Flux only</td>
  </tr>

  <tr>
    <td>BeyondNStd</td>
    <td>Fraction of observations beyond $n\sigma _{m}$ from the mean magnitude $\langle m\rangle $:
<p align="center">$\displaystyle \frac{\sum _{i} I_{|m-\langle m\rangle | >n\sigma _{m}} (m_{i} )}{N},$</p> where $I-$ an indicator function</td>
    <td align="center">2</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
  </tr>

  <tr>
    <td><i>ColorOfMedian</i> <br>(experimental)</td>
    <td>Magnitude difference between medians of two bands</td>
    <td align="center">2</td>
    <td align="center">1</td>
    <td>Magn only</td>
  </tr>

   <tr>
    <td>Cusum</td>
    <td>A range of cumulative sums:
<p align="center">$\displaystyle \max(S) -\min(S),$</p> where $S_{j} \equiv \frac{1}{N\sigma _{m}}\sum\limits _{i=0}^{j} (m_{i} -\langle m\rangle )$, $j\in \{1..N-1\}$</td>
    <td align="center">2</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
  </tr>

  <tr>
    <td>Eta</td>
    <td>Von Neummann $\eta $:
<p align="center">$\displaystyle \eta \equiv \frac{1}{(N-1)\sigma _{m}^{2}}\sum\limits _{i=0}^{N-2} (m_{i+1} -m_{i} )^{2}$</p></td>
    <td align="center">2</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
  </tr>

  <tr>
    <td>EtaE</td>
    <td>Modernisation of <b>Eta</b> for unevenly time series:
<p align="center">$\displaystyle \eta ^{e} \equiv \frac{(t_{N-1} -t_{0} )^{2}}{(N-1)^{3}}\frac{\sum\limits_{i=0}^{N-2}\left(\frac{m_{i+1} -m_{i}}{t_{i+1} -t_{i}}\right)^{2}}{\sigma _{m}^{2}}$</p></td>
    <td align="center">2</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
  </tr>

  <tr>
    <td>ExcessVariance</td>
    <td>Measure of the variability amplitude:
<p align="center">$\displaystyle \frac{\sigma _{m}^{2} -\langle \delta ^{2} \rangle }{\langle m\rangle ^{2}},$</p> where $\langle \delta ^{2} \rangle -$ mean squared error</td>
    <td align="center">2</td>
    <td align="center">1</td>
    <td>Flux only</td>
  </tr>

  <tr>
    <td><i>FluxNNotDetBeforeFd</i><br>(experimental)</td>
    <td>Number of non-detections before the first detection</td>
    <td align="center">2</td>
    <td align="center">1</td>
    <td>Flux only</td>
  </tr>

  <tr>
    <td>InterPercentileRange</td>
    <td><p align="center">$\displaystyle Q(1-p)-Q(p),$</p> where $Q(n)$ and $Q(d)-$ $n$-th and $d$-th quantile of magnitude sample</td>
    <td align="center">1</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
  </tr>

  <tr>
    <td>Kurtosis</td>
    <td>Excess kurtosis of magnitude:
<p align="center">$\displaystyle \frac{N(N+1)}{(N-1)(N-2)(N-3)}\frac{\sum _{i} (m_{i} -\langle m\rangle )^{4}}{\sigma _{m}^{2}} -3\frac{(N+1)^{2}}{(N-2)(N-3)}$</p></td>
    <td align="center">4</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
  </tr>

  <tr>
    <td>LinearFit</td>
    <td>The slope, its error and reduced $\chi ^{2}$ of the light curve in the linear fit of a magnitude light curve with respect to the observation error $\{\delta _{i}\}$:
<p align="center">$\displaystyle m_{i} \ =\ c\ +\ \text{slope} \ t_{i} \ +\ \delta _{i} \varepsilon _{i} ,$</p> where $c$ is a constant, $\{\varepsilon _{i}\}$ are standard distributed random variables</td>
    <td align="center">3</td>
    <td align="center">3</td>
    <td>Flux or magn</td>
  </tr>

  <tr>
    <td>LinearTrend</td>
    <td>The slope and its error of the light curve in the linear fit of a magnitude light curve without respect to the observation error $\{\delta _{i}\}$:
<p align="center">$\displaystyle m_{i} \ =\ c\ +\ \text{slope} \ t_{i} \ +\ \Sigma \varepsilon _{i} ,$</p> where $c$ and $\Sigma$ are constants, $\{\varepsilon _{i}\}$  are standard distributed random variables.</td>
    <td align="center">2</td>
    <td align="center">2</td>
    <td>Flux or magn</td>
  </tr>

  <tr>
    <td><i>MagnitudeNNotDetBeforeFd</i><br>(experimental)</td>
    <td>Number of non-detections before the first detection</td>
    <td align="center">2</td>
    <td align="center">1</td>
    <td>Magn only</td>
  </tr>

  <tr>
    <td>MagnitudePercentageRatio</td>
    <td>Magnitude percentage ratio:
<p align="center">$\displaystyle \frac{Q(1-n)-Q(n)}{Q(1-d)-Q(d)}$</p></td>
    <td align="center">1</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
  </tr>

  <tr>
    <td>MaximumSlope</td>
    <td>Maximum slope between two sub-sequential observations:
<p align="center">$\displaystyle \max_{i=0\dotsc N-2}\left| \frac{m_{i+1} -m_{i}}{t_{i+1} -t_{i}}\right|$</p></td>
    <td align="center">2</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
  </tr>

  <tr>
    <td>Mean</td>
    <td>Mean magnitude:
<p align="center">$\displaystyle \langle m\rangle =\frac{1}{N}\sum\limits _{i} m_{i}$</p></td>
    <td align="center">1</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
  </tr>

  <tr>
    <td>MeanVariance</td>
    <td>Standard deviation to mean ratio:
<p align="center">$\displaystyle \frac{\sigma _{m}}{\langle m\rangle }$</p></td>
    <td align="center">2</td>
    <td align="center">1</td>
    <td>Flux only</td>
  </tr>

  <tr>
    <td>Median</td>
    <td>Median magnitude
    <td align="center">1</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
  </tr>

  <tr>
    <td>MedianAbsoluteDeviation</td>
    <td>Median of the absolute value of the difference between magnitude and its median:
<p align="center">$\displaystyle \mathrm{Median} (|m_{i} -\mathrm{Median} (m)|)$</p></td>
    <td align="center">1</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
  </tr>

  <tr>
    <td>MedianBufferRangePercentage</td>
    <td>Fraction of points within <p align="center">$\displaystyle \mathrm{Median} (m)\pm q\times (\max (m)-\min (m))/2$</p></td>
    <td align="center">1</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
  </tr>

  <tr>
    <td>OtsuSplit</td>
    <td>Difference of subset means, standard deviation of the lower subset, standard deviation of the upper
subset and lower-to-all observation count ratio for two subsets of magnitudes obtained by Otsu's method split.
<br>
<br>
Otsu's method is used to perform automatic thresholding. The algorithm returns a single threshold that separate values into two classes. This threshold is determined by minimizing intra-class intensity variance $\sigma^2_{W}=w_0\sigma^2_0+w_1\sigma^2_1$, or equivalently, by maximizing inter-class variance $\sigma^2_{B}=w_0 w_1 (\mu_1-\mu_0)^2$. There can be more than one extremum. In this case, the algorithm returns the minimum threshold.
   </td>
    <td align="center">2</td>
    <td align="center">4</td>
    <td>Flux or magn</td>
  </tr>

  <tr>
    <td>PercentAmplitude</td>
    <td>Maximum deviation of magnitude from its median:
<p align="center">$\displaystyle \max_{i} |m_{i} \ -\ \text{Median}( m) |$</p></td>
    <td align="center">1</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
  </tr>

  <tr>
    <td>PercentDifferenceMagnitudePercentile</td>
    <td>Ratio of $p$-th inter-percentile range to the median:
<p align="center">$\displaystyle \frac{Q( 1-p) -Q( p)}{\text{Median}( m)}$</p></td>
    <td align="center">1</td>
    <td align="center">1</td>
    <td>Flux only</td>
  </tr>

  <tr>
    <td><i>RainbowFit</i><br>(experimental)</td>
    <td>Seven fit parameters and goodness of fit (reduced $\chi ^{2}$). The Rainbow method is developed and detailed here : https://arxiv.org/abs/2310.02916). This implementation is suited for transient objects. Bolometric flux and temperature functions are customizable, by default Bazin and logistic functions are used:
<p align="center">$\displaystyle F_{\nu}(t, \nu) = \frac{\pi\,B\left(T(t),\nu\right)}{\sigma_\mathrm{SB}\,T(t)^{4}} \times F_\mathrm{bol}(t),$</p> where $F_{\nu}(t, \nu)-$ flux observation at a given wavelength</td>
    <td align="center">6</td>
    <td align="center">1</td>
    <td>Flux only</td>
  </tr>

  <tr>
    <td>ReducedChi2</td>
    <td>Reduced $\chi ^{2}$ of magnitude measurements:
<p align="center">$\displaystyle \frac{1}{N-1}\sum _{i}\left(\frac{m_{i} -\overline{m}}{\delta _{i}}\right)^{2} ,$</p> where $\overline{m} -$ weighted mean magnitude</td>
    <td align="center">2</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
  </tr>

  <tr>
    <td><i>Roms</i><br>(Experimental)</td>
    <td>Robust median statistic: <p align="center">$\displaystyle \frac1{N-1} \sum_{i=0}^{N-1} \frac{|m_i - \mathrm{median}(m_i)|}{\sigma_i}$</p></td>
    <td align="center">2</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
  </tr>

  <tr>
    <td>Skew</td>
    <td>Skewness of magnitude:
<p align="center">$\displaystyle \frac{N}{(N-1)(N-2)}\frac{\sum _{i} (m_{i} -\langle m\rangle )^{3}}{\sigma _{m}^{3}}$</p></td>
    <td align="center">3</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
  </tr>

  <tr>
    <td>StandardDeviation</td>
    <td>Standard deviation of magnitude:
<p align="center">$\displaystyle \sigma _{m} \equiv \sqrt{\sum _{i} (m_{i} -\langle m\rangle )^{2} /(N-1)}$</p></td>
    <td align="center">2</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
  </tr>

  <tr>
    <td>StetsonK</td>
    <td><b>Stetson K</b> coefficient described light curve shape:
<p align="center">$\displaystyle \frac{\sum _{i}\left| \frac{m_{i} -\langle m\rangle }{\delta _{i}}\right| }{\sqrt{N\ \chi ^{2}}}$</p></td>
    <td align="center">2</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
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
  </tr>

  <tr>
    <td>WeightedMean</td>
    <td>Weighted mean magnitude:
<p align="center">$\displaystyle \overline{m} \equiv \frac{\sum _{i} m_{i} /\delta _{i}^{2}}{\sum _{i} 1/\delta _{i}^{2}}$</p></td>
    <td align="center">1</td>
    <td align="center">1</td>
    <td>Flux or magn</td>
  </tr>

</table>

### Meta-features

Meta-features can accept other feature extractors and apply them to pre-processed data.

#### Periodogram

This feature transforms time-series data into the Lomb-Scargle periodogram, providing an estimation of the power
spectrum. The peaks argument corresponds to the number of the most significant spectral density peaks to return. For
each peak, its period and "signal-to-noise" ratio are returned.

$$
\text{signal to noise of peak} \equiv \frac{P(\omega_\mathrm{peak}) - \langle P(\omega) \rangle}{\sigma\_{P(\omega)}}
$$

The optional features argument accepts a list of additional feature evaluators, which are applied to the power spectrum:
frequency is passed as "time," power spectrum is passed as "magnitude," and no uncertainties are set.

#### Bins

Binning time series to bins with width $\mathrm{window}$ with respect to some $\mathrm{offset}$.
$j-th$ bin boundaries
are $[j \cdot \mathrm{window} + \mathrm{offset}; (j + 1) \cdot \mathrm{window} + \mathrm{offset}]$.

Binned time series is defined by
$$t_j^* = (j + \frac12) \cdot \mathrm{window} + \mathrm{offset},$$
$$m_j^* = \frac{\sum{m_i / \delta_i^2}}{\sum{\delta_i^{-2}}},$$
$$\delta_j^* = \frac{N_j}{\sum{\delta_i^{-2}}},$$
where $N_j$ is a number of sampling observations and all sums are over observations inside considering bin.

### Multi-band features

As of v0.8, experimental extractors (see below), support multi-band light-curve inputs.

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

Rainbow ([Russeil+23](https://arxiv.org/abs/2310.02916)) is a black-body parametric model for transient light curves.
By default, it uses Bazin function as a model for bolometric flux evolution and a logistic function for the temperature
evolution.
The user may customize the model by providing their own functions for bolometric flux and temperature evolution.
This example demonstrates the reconstruction of a synthetic light curve with this model.
`RainbowFit` requires `iminuit` package.

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
lum = amplitude * np.exp(-(t - reference_time) / fall_time) / (1.0 + np.exp(-(t - reference_time) / rise_time))

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

Note, that while we don't use precise physical constant values to generate the data, `RainbowFit` uses CODATA 2018
values.

### Experimental extractors

From the technical point of view the package consists of two parts: a wrapper
for [`light-curve-feature` Rust crate](https://crates.io/crates/light-curve-feature) (`light_curve_ext` sub-package) and
pure Python sub-package `light_curve_py`.
We use the Python implementation of feature extractors to test Rust implementation and to implement new experimental
extractors.
Please note, that the Python implementation is much slower for most of the extractors and doesn't provide the same
functionality as the Rust implementation.
However, the Python implementation provides some new feature extractors you can find useful.

You can manually use extractors from both implementations:

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

This should print a warning about experimental status of the Python class

### Benchmarks

You can run all benchmarks from the Python project folder
with `python3 -mpytest --benchmark-enable tests/test_w_bench.py`, or with slow benchmarks
disabled `python3 -mpytest -m "not (nobs or multi)" --benchmark-enable tests/test_w_bench.py`.

Here we benchmark the Rust implementation (`rust`) versus [`feets`](https://feets.readthedocs.io/en/latest/) package and
our own Python implementation (`lc_py`) for a light curve having n=1000 observations.

![Benchmarks, Rust is much faster](https://github.com/light-curve/light-curve-python/raw/readme-benchs/light-curve/.readme/benchplot_v2.png)

The plot shows that the Rust implementation of the package outperforms other ones by a factor of 1.5—50.
This allows to extract a large set of "cheap" features well under one ms for n=1000.
The performance of parametric fits (`BazinFit` and `VillarFit`) and `Periodogram` depend on their parameters, but the
typical timescale of feature extraction including these features is 20—50 ms for few hundred observations.

![Benchmark for different number of observations](https://github.com/light-curve/light-curve-python/raw/readme-benchs/light-curve/.readme/nobs_bench_v2.png)

Benchmark results of several features for both the pure-Python and Rust implementations of the "light-curve" package, as
a function of the number of observations in a light curve. Both the x-axis and y-axis are on a logarithmic scale.

![Benchmark for multithreading and multiprocessing](https://github.com/light-curve/light-curve-python/raw/readme-benchs/light-curve/.readme/multi_bench_v2.png)

Processing time per a single light curve for extraction of features subset presented in first benchmark versus the
number of CPU cores used. The dataset consists of 10,000 light curves with 1,000 observations in each.

See benchmarks' descriptions in more details
in ["Performant feature extraction for photometric time series"](https://arxiv.org/abs/2302.10837).

## dm-dt map

Class `DmDt` provides dm–dt mapper (based
on [Mahabal et al. 2011](https://ui.adsabs.harvard.edu/abs/2011BASI...39..387M/abstract), [Soraisam et al. 2020](https://ui.adsabs.harvard.edu/abs/2020ApJ...892..112S/abstract)).
It is a Python wrapper for [`light-curve-dmdt` Rust crate](https://crates.io/crates/light-curve-dmdt).

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

## Citation

If you found this project useful for your research please
cite [Malanchev et al., 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.502.5147M/abstract)

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
