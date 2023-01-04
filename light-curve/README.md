# `light-curve` processing toolbox for Python

The Python wrapper for Rust [`light-curve-feature`](https://github.com/light-curve/light-curve-feature) and [`light-curve-dmdt`](https://github.com/light-curve/light-curve-dmdt) packages which gives a collection of high-performant time-series feature extractors.

[![PyPI version](https://badge.fury.io/py/light-curve.svg)](https://pypi.org/project/light-curve/)
![testing](https://github.com/light-curve/light-curve-python/actions/workflows/test.yml/badge.svg)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/light-curve/light-curve-python/master.svg)](https://results.pre-commit.ci/latest/github/light-curve/light-curve-python/master)

## Installation

```sh
python3 -mpip install light-curve
```

We also provide `light-curve-python` package which is just an "alias" to the main `light-curve` package.

Minimum supported Python version is 3.7.
We provide binary wheels via [PyPi](https://pypi.org/project/light-curve/) for number of platforms and architectures, both for CPython and PyPy.

### Support matrix

| Arch \ OS   | Linux glibc | Linux musl                     | macOS                                                          | Windows                                                                |
| ----------- |-------------|--------------------------------|----------------------------------------------------------------|------------------------------------------------------------------------|
| **x86-64**  | wheel (MKL) | wheel (MKL)                    | wheel                                                          | not tested https://github.com/light-curve/light-curve-python/issues/12 |
| **i686**    | src         | src                            | —                                                              | not tested                                                             |
| **aarch64** | wheel       | wheel                          | src https://github.com/light-curve/light-curve-python/issues/5 | not tested                                                             |
| **ppc64le** | wheel       | not tested (no Rust toolchain) | —                                                              | —                                                                      |

- "wheel": binary wheel is available on pypi.org, local building is not required for the platform, the only pre-requirement is a recent `pip` version. For Linux x86-64 we provide binary wheels built with Intel MKL for better periodogram perfromance, which is not a default build option.
- "src": the package is confirmed to be built and pass unit tests locally, but testing and package building is not supported by CI. It is required to have the [GNU scientific library (GSL)](https://www.gnu.org/software/gsl/) v2.1+ and the [Rust toolchain](https://rust-lang.org) v1.57+ to install it via `pip install`.
- "not tested": building from the source code is not tested, please report us building status via issue/PR/email.

We build aarch64 macOS 12.0+ Python 3.8+ wheels locally and submit them running this command in `light-curve` directory:
```
rm -rf ./wheelhouse
CIBW_BUILD='cp3*-macosx_arm64' CIBW_ENVIRONMENT="MACOSX_DEPLOYMENT_TARGET=12.0 MATURIN_PEP517_ARGS='--locked --no-default-features --features fftw-source,gsl'" CIBW_BEFORE_ALL='curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain stable -y; brew install gsl' python3 -mcibuildwheel --platform macos
twine upload wheelhouse/*.whl
```

## Feature evaluators

Most of the classes implement various feature evaluators useful for light-curve based
astrophysical source classification and characterisation.

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

If you confident in your inputs you could use `sorted = True` (`t` is in ascending order)
and `check = False` (no NaNs in inputs, no infs in `t` or `m`) for better performance.
Note that if your inputs are not valid and are not validated by
`sorted=None` and `check=True` (default values) then all kind of bad things could happen.

Print feature classes list
```python
import light_curve as lc

print([x for x in dir(lc) if hasattr(getattr(lc, x), "names")])
```

Read feature docs
```python
import light_curve as lc

help(lc.BazinFit)
```

### Experimental extractors

From the technical point of view the package consists of two parts: a wrapper for [`light-curve-feature` Rust crate](https://crates.io/crates/light-curve-feature) (`light_curve_ext` sub-package) and pure Python sub-package `light_curve_py`.
We use the Python implementation of feature extractors to test Rust implementation and to implement new experimental extractors.
Please note, that the Python implementation is much slower for the most of the extractors and doesn't provide the same functionality as the Rust implementation.
However, the Python implementation provides some new feature extractors you can find useful.

You can manually use extractors from both implementations:

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

### Available features

See the complite list of evailable feature evaluators and documentation in [`light-curve-feature` Rust crate docs](https://docs.rs/light-curve-feature/latest/light_curve_feature/features/index.html).

<table>
  <tr>
    <th>Feature name</th>
    <th>Description</th>
    <th>Min data points</th>
    <th>Features number</th>
  </tr>
  <tr>
    <td>Amplitude</td>
    <td>Half amplitude of magnitude: <p align="center">$\displaystyle \frac{\max (m)-\min (m)}{2}$</p></td>
    <td align="center">1</td>
    <td align="center">1</td>
  </tr>

  <tr>
    <td>AndersonDarlingNormal</td>
    <td>Unbiased Anderson–Darling normality test statistic:
<p align="left">$\displaystyle \left( 1+\frac{4}{N} -\frac{25}{N^{2}}\right) \times$<p>
<p align="right">$\times \left( -N-\frac{1}{N}\sum\limits_{i=0}^{N-1} (2i+1)\ln \Phi _{i} +(2(N-i)-1)\ln (1-\Phi _{i} )\right) ,$<p> where $\Phi _{i\ } \equiv \Phi (( m_{i} \ -\ \langle m\rangle ) /\sigma _{m})$ is the commutative distribution function of the standard normal distribution, $N-$ the number of observations, $\langle m\rangle -$ mean magnitude and $\sigma _{m} =\sqrt{\sum\limits_{i=0}^{N-1}( m_{i} -\langle m\rangle )^{2} /( N-1) \ }$ is the magnitude standard deviation</td>
    <td align="center">4</td>
    <td align="center">1</td>
  </tr>

  <tr>
    <td>BazinFit</td>
    <td>Five fit parameters and goodness of fit (reduced $\chi ^{2}$ of the Bazin function developed for core-collapsed supernovae:
<p align="center">$\displaystyle f(t)=A\frac{\mathrm{e}^{-(t-t_{0} )/\tau _{fall}}}{1+\mathrm{e}^{-(t-t_{0} )/\tau _{rise}}} +B,$</p> where $f(t)-$ flux observation</td>
    <td align="center">6</td>
    <td align="center">1</td>
  </tr>

  <tr>
    <td>BeyondNStd</td>
    <td>Fraction of observations beyond $n\sigma _{m}$ from the mean magnitude $\langle m\rangle $:
<p align="center">$\displaystyle \frac{\sum _{i} I_{|m-\langle m\rangle | >n\sigma _{m}} (m_{i} )}{N},$</p> where $I-$ an indicator function</td>
    <td align="center">2</td>
    <td align="center">1</td>
  </tr>

   <tr>
    <td>Cusum</td>
    <td>A range of cumulative sums:
<p align="center">$\displaystyle \max(S) -\min(S),$</p> where $S_{j} \equiv \frac{1}{N\sigma _{m}}\sum\limits _{i=0}^{j} (m_{i} -\langle m\rangle )$, $j\in \{1..N-1\}$</td>
    <td align="center">2</td>
    <td align="center">1</td>
  </tr>

  <tr>
    <td>Eta</td>
    <td>Von Neummann $\eta $:
<p align="center">$\displaystyle \eta \equiv \frac{1}{(N-1)\sigma _{m}^{2}}\sum\limits _{i=0}^{N-2} (m_{i+1} -m_{i} )^{2}$</p></td>
    <td align="center">2</td>
    <td align="center">1</td>
  </tr>

  <tr>
    <td>EtaE</td>
    <td>Modernisation of <b>Eta</b> for unevenly time series:
<p align="center">$\displaystyle \eta ^{e} \equiv \frac{(t_{N-1} -t_{0} )^{2}}{(N-1)^{3}}\frac{\sum\limits_{i=0}^{N-2}\left(\frac{m_{i+1} -m_{i}}{t_{i+1} -t_{i}}\right)^{2}}{\sigma _{m}^{2}}$</p></td>
    <td align="center">2</td>
    <td align="center">1</td>
  </tr>

  <tr>
    <td>ExcessVariance</td>
    <td>Measure of the variability amplitude:
<p align="center">$\displaystyle \frac{\sigma _{m}^{2} -\langle \delta ^{2} \rangle }{\langle m\rangle ^{2}},$</p> where $\langle \delta ^{2} \rangle -$ mean squared error</td>
    <td align="center">2</td>
    <td align="center">1</td>
  </tr>

  <tr>
    <td>InterPercentileRange</td>
    <td><p align="center">$\displaystyle Q(1-p)-Q(p),$</p> where $Q(n)$ and $Q(d)-$ $n$-th and $d$-th quantile of magnitude sample</td>
    <td align="center">1</td>
    <td align="center">1</td>
  </tr>

  <tr>
    <td>Kurtosis</td>
    <td>Excess kurtosis of magnitude:
<p align="center">$\displaystyle \frac{N(N+1)}{(N-1)(N-2)(N-3)}\frac{\sum _{i} (m_{i} -\langle m\rangle )^{4}}{\sigma _{m}^{2}} -3\frac{(N+1)^{2}}{(N-2)(N-3)}$</p></td>
    <td align="center">4</td>
    <td align="center">1</td>
  </tr>

  <tr>
    <td>LinearFit</td>
    <td>The slope, its error and reduced $\chi ^{2}$ of the light curve in the linear fit of a magnitude light curve with respect to the observation error $\{\delta _{i}\}$:
<p align="center">$\displaystyle m_{i} \ =\ c\ +\ \text{slope} \ t_{i} \ +\ \delta _{i} \varepsilon _{i} ,$</p> where $c$ is a constant, $\{\varepsilon _{i}\}$ are standard distributed random variables</td>
    <td align="center">3</td>
    <td align="center">3</td>
  </tr>

  <tr>
    <td>LinearTrend</td>
    <td>The slope and its error of the light curve in the linear fit of a magnitude light curve without respect to the observation error $\{\delta _{i}\}$:
<p align="center">$\displaystyle m_{i} \ =\ c\ +\ \text{slope} \ t_{i} \ +\ \Sigma \varepsilon _{i} ,$</p> where $c$ and $\Sigma$ are constants, $\{\varepsilon _{i}\}$  are standard distributed random variables.</td>
    <td align="center">2</td>
    <td align="center">2</td>
  </tr>

  <tr>
    <td>MagnitudePercentageRatio</td>
    <td>Magnitude percentage ratio:
<p align="center">$\displaystyle \frac{Q(1-n)-Q(n)}{Q(1-d)-Q(d)}$</p></td>
    <td align="center">1</td>
    <td align="center">1</td>
  </tr>

  <tr>
    <td>MaximumSlope</td>
    <td>Maximum slope between two sub-sequential observations:
<p align="center">$\displaystyle \max_{i=0\dotsc N-2}\left| \frac{m_{i+1} -m_{i}}{t_{i+1} -t_{i}}\right|$</p></td>
    <td align="center">2</td>
    <td align="center">1</td>
  </tr>

  <tr>
    <td>Mean</td>
    <td>Mean magnitude:
<p align="center">$\displaystyle \langle m\rangle =\frac{1}{N}\sum\limits _{i} m_{i}$</p></td>
    <td align="center">1</td>
    <td align="center">1</td>
  </tr>

  <tr>
    <td>MeanVariance</td>
    <td>Standard deviation to mean ratio:
<p align="center">$\displaystyle \frac{\sigma _{m}}{\langle m\rangle }$</p></td>
    <td align="center">2</td>
    <td align="center">1</td>
  </tr>

  <tr>
    <td>Median</td>
    <td>Median magnitude
    <td align="center">1</td>
    <td align="center">1</td>
  </tr>

  <tr>
    <td>MedianAbsoluteDeviation</td>
    <td>Median of the absolute value of the difference between magnitude and its median:
<p align="center">$\displaystyle \mathrm{Median} (|m_{i} -\mathrm{Median} (m)|)$</p></td>
    <td align="center">1</td>
    <td align="center">1</td>
  </tr>

  <tr>
    <td>MedianBufferRangePercentage</td>
    <td><p align="center">$\displaystyle \mathrm{Median} (m)\pm q\times (\max (m)-\min (m))/2$</p></td>
    <td align="center">1</td>
    <td align="center">1</td>
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
  </tr>

  <tr>
    <td>PercentAmplitude</td>
    <td>Maximum deviation of magnitude from its median:
<p align="center">$\displaystyle \max_{i} |m_{i} \ -\ \text{Median}( m) |$</p></td>
    <td align="center">1</td>
    <td align="center">1</td>
  </tr>

  <tr>
    <td>PercentDifferenceMagnitudePercentile</td>
    <td>Ratio of $p$-th inter-percentile range to the median:
<p align="center">$\displaystyle \frac{Q( 1-p) -Q( p)}{\text{Median}( m)}$</p></td>
    <td align="center">1</td>
    <td align="center">1</td>
  </tr>

  <tr>
    <td>ReducedChi2</td>
    <td>Reduced $\chi ^{2}$ of magnitude measurements:
<p align="center">$\displaystyle \frac{1}{N-1}\sum _{i}\left(\frac{m_{i} -\overline{m}}{\delta _{i}}\right)^{2} ,$</p> where $\overline{m} -$ weighted mean magnitude</td>
    <td align="center">2</td>
    <td align="center">1</td>
  </tr>

  <tr>
    <td>Skew</td>
    <td>Skewness of magnitude:
<p align="center">$\displaystyle \frac{N}{(N-1)(N-2)}\frac{\sum _{i} (m_{i} -\langle m\rangle )^{3}}{\sigma _{m}^{3}}$</p></td>
    <td align="center">3</td>
    <td align="center">1</td>
  </tr>

  <tr>
    <td>StandardDeviation</td>
    <td>Standard deviation of magnitude:
<p align="center">$\displaystyle \sigma _{m} \equiv \sqrt{\sum _{i} (m_{i} -\langle m\rangle )^{2} /(N-1)}$</p></td>
    <td align="center">2</td>
    <td align="center">1</td>
  </tr>

  <tr>
    <td>StetsonK</td>
    <td><b>Stetson K</b> coefficient described light curve shape:
<p align="center">$\displaystyle \frac{\sum _{i}\left| \frac{m_{i} -\langle m\rangle }{\delta _{i}}\right| }{\sqrt{N\ \chi ^{2}}}$</p></td>
    <td align="center">2</td>
    <td align="center">1</td>
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
  </tr>

  <tr>
    <td>WeightedMean</td>
    <td>Weighted mean magnitude:
<p align="center">$\displaystyle \overline{m} \equiv \frac{\sum _{i} m_{i} /\delta _{i}^{2}}{\sum _{i} 1/\delta _{i}^{2}}$</p></td>
    <td align="center">1</td>
    <td align="center">1</td>
  </tr>

</table>

### Meta-features
Meta-features can accept another feature extractors and apply them to pre-processed data.

#### Periodogram

A number of features based on Lomb–Scargle periodogram as it was an evenly separeated uncertancy-less lime series.
Periodogram $P(\omega)$ is an estimate of spectral density of unevenly time series.
`Periodogram`'s `peaks` argument corresponds to a number of the most significant spectral density peaks to return.
For each peak its period and "signal to noise" ratio is returned.

$$
\text{signal to noise of peak} \equiv \frac{P(\omega_\mathrm{peak}) - \langle P(\omega) \rangle}{\sigma\_{P(\omega)}}
$$

`features` argument accepts a list of additional feature evaluators.

#### Bins

Binning time series to bins with width $\mathrm{window}$ with respect to some $\mathrm{offset}$.
$j-th$ bin boundaries are $[j \cdot \mathrm{window} + \mathrm{offset}; (j + 1) \cdot \mathrm{window} + \mathrm{offset}]$.

Binned time series is defined by
$$t_j^* = (j + \frac12) \cdot \mathrm{window} + \mathrm{offset},$$
$$m_j^* = \frac{\sum{m_i / \delta_i^2}}{\sum{\delta_i^{-2}}},$$
$$\delta_j^* = \frac{N_j}{\sum{\delta_i^{-2}}},$$
where $N_j$ is a number of sampling observations and all sums are over observations inside considering bin.


### Benchmarks

We benchmark the Rust implementation (`rust`) versus [`feets`](https://feets.readthedocs.io/en/latest/) package and our own Python implementation (`lc_py`) for a light curve having n=1000 observations.

![Benchmarks, Rust is much faster](https://github.com/light-curve/light-curve-python/raw/readme-benchs/light-curve/.readme/benchplot.png)

The plot shows that the Rust implementation of the package outperforms other ones by a factor of 1.5—50.
This allows to extract a large set of "cheap" features well under one ms for n=1000.
The performance of parametric fits (`BazinFit` and `VillarFit`) and `Periodogram` depend on their parameters, but the typical timescale of feature extraction including these features is 20—50 ms for few hundred observations.

## dm-dt map

Class `DmDt` provides dm–dt mapper (based on [Mahabal et al. 2011](https://ui.adsabs.harvard.edu/abs/2011BASI...39..387M/abstract), [Soraisam et al. 2020](https://ui.adsabs.harvard.edu/abs/2020ApJ...892..112S/abstract)). It is a Python wrapper for [`light-curve-dmdt` Rust crate](https://crates.io/crates/light-curve-dmdt).

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

If you found this project useful for your research please cite [Malanchev et al., 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.502.5147M/abstract)

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
