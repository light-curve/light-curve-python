# `light-curve` processing toolbox for Python

The Python wrapper for Rust [`light-curve-feature`](https://github.com/light-curve/light-curve) and [`light-curve-dmdt`](https://github.com/light-curve/light-curve) packages which gives a collection of high-performant time-series feature extractors.

[![PyPI version](https://badge.fury.io/py/light-curve.svg)](https://pypi.org/project/light-curve/)
![testing](https://github.com/light-curve/light-curve-python/actions/workflows/test.yml/badge.svg)
![publishing](https://github.com/light-curve/light-curve-python/actions/workflows/publish.yml/badge.svg)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/light-curve/light-curve-python/master.svg)](https://results.pre-commit.ci/latest/github/light-curve/light-curve-python/master)

## Installation

```sh
python3 -mpip install light-curve
```

Minimum supported Python version is 3.6.
The package is tested on Linux (x86-64, aarch64, ppc64) and macOS (x86-64).
Pre-built wheels for these platforms are available on [pypi.org](https://pypi.org/project/light-curve/#files), other systems are required to have [GNU scientific library (GSL)](https://www.gnu.org/software/gsl/) v2.1+ and the [Rust](https://rust-lang.org) toolchain v1.56+ to build and install the package.
You can use [`rustup` script](https://rustup.rs) to get the most recent Rust toolchain.

Also could find `light-curve-python` package which is just an "alias" to the main `light-curve` package.

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

### Citation

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
