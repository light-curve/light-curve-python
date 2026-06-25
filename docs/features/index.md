# Feature extractors

`light_curve` provides 40+ hand-crafted feature extractors for astrophysical light curves.
All share a common interface: callable objects with `.names` and `.descriptions` attributes.

```python
import light_curve as licu
import numpy as np

t = np.array([0.0, 1.2, 3.5, 7.1, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0])
m = np.array([15.1, 15.3, 14.9, 15.0, 15.2, 15.1, 14.8, 15.3, 15.0, 15.1])
err = np.array([0.05] * 10)
band = np.tile(["g", "r"], 5)

ext = licu.Extractor(licu.Chi2Pvar(), licu.BeyondNStd(nstd=1), licu.LinearFit(bands=["g", "r"]))
result = ext(t, m, err, band)
print(dict(zip(ext.names, result)))
```

Use [`Extractor`](api/meta.md#light_curve.Extractor) to combine multiple features into a single callable,
or call each class directly for a single feature.
The `.many()` method processes a list of light curves in batch with reduced Python–Rust overhead.

See the [API reference](api/variability.md) for full signatures, parameters, and equations.

---

## Feature table

### Variability

*Fast: all 26 features combined ~100 µs on 1,000 observations, single band.*

| Feature | Description | Outputs |
|---------|-------------|---------|
| [`Amplitude`](api/variability.md#light_curve.Amplitude) | Half peak-to-peak amplitude | 1 |
| [`AndersonDarlingNormal`](api/variability.md#light_curve.AndersonDarlingNormal) | Unbiased Anderson–Darling normality test statistic | 1 |
| [`BeyondNStd`](api/variability.md#light_curve.BeyondNStd) | Fraction of observations beyond \(n\,\sigma_m\) from \(\langle m \rangle\) | 1 |
| [`Chi2Pvar`](api/variability.md#light_curve.Chi2Pvar) | Probability of variability from the \(\chi^2\) test | 1 |
| [`Cusum`](api/variability.md#light_curve.Cusum) | Range of cumulative sums | 1 |
| [`Eta`](api/variability.md#light_curve.Eta) | Von Neumann \(\eta\) | 1 |
| [`EtaE`](api/variability.md#light_curve.EtaE) | \(\eta^e\), adapted for unevenly sampled time series | 1 |
| [`ExcessVariance`](api/variability.md#light_curve.ExcessVariance) | Measure of intrinsic variability amplitude | 1 |
| [`InterPercentileRange`](api/variability.md#light_curve.InterPercentileRange) | \(Q(1-p) - Q(p)\) inter-percentile range | 1 |
| [`Kurtosis`](api/variability.md#light_curve.Kurtosis) | Excess kurtosis of magnitude | 1 |
| [`LaflerKinmanStringLength`](api/variability.md#light_curve.LaflerKinmanStringLength) | Lafler–Kinman string-length smoothness statistic | 1 |
| [`MaximumSlope`](api/variability.md#light_curve.MaximumSlope) | Maximum slope between consecutive observations | 1 |
| [`Mean`](api/variability.md#light_curve.Mean) | Mean magnitude | 1 |
| [`MeanVariance`](api/variability.md#light_curve.MeanVariance) | Standard deviation to mean ratio | 1 |
| [`Median`](api/variability.md#light_curve.Median) | Median magnitude | 1 |
| [`MedianAbsoluteDeviation`](api/variability.md#light_curve.MedianAbsoluteDeviation) | Median of \(\lvert m_i - \mathrm{median}(m) \rvert\) | 1 |
| [`MedianBufferRangePercentage`](api/variability.md#light_curve.MedianBufferRangePercentage) | Fraction of points within \(q \times \mathrm{amplitude}\) of median | 1 |
| [`OtsuSplit`](api/variability.md#light_curve.OtsuSplit) | Otsu thresholding: bimodality measure (subset means, std devs, fraction) | 4 |
| [`PercentAmplitude`](api/variability.md#light_curve.PercentAmplitude) | Maximum deviation of magnitude from its median | 1 |
| [`PercentDifferenceMagnitudePercentile`](api/variability.md#light_curve.PercentDifferenceMagnitudePercentile) | Ratio of inter-percentile range to median | 1 |
| [`ReducedChi2`](api/variability.md#light_curve.ReducedChi2) | Reduced \(\chi^2\) of magnitude measurements | 1 |
| [`Roms`](api/variability.md#light_curve.Roms) | Robust median statistic *(experimental)* | 1 |
| [`Skew`](api/variability.md#light_curve.Skew) | Skewness \(G_1\) of magnitude | 1 |
| [`StandardDeviation`](api/variability.md#light_curve.StandardDeviation) | Standard deviation \(\sigma_m\) of magnitude | 1 |
| [`StetsonK`](api/variability.md#light_curve.StetsonK) | Stetson \(K\) light-curve shape coefficient | 1 |
| [`WeightedMean`](api/variability.md#light_curve.WeightedMean) | Inverse-variance weighted mean magnitude | 1 |

### Linear trend

*Fast: both features combined ~10 µs on 1,000 observations, single band.*

| Feature | Description | Outputs |
|---------|-------------|---------|
| [`LinearFit`](api/linear.md#light_curve.LinearFit) | Slope, its error, and reduced \(\chi^2\) of the weighted linear fit | 3 |
| [`LinearTrend`](api/linear.md#light_curve.LinearTrend) | Slope, its error, and noise level of the unweighted linear fit | 3 |

### Time sampling

*Fast: all 6 features combined ~5 µs on 1,000 observations, single band.*

| Feature | Description | Outputs |
|---------|-------------|---------|
| [`Duration`](api/time.md#light_curve.Duration) | Total time span of the light curve | 1 |
| [`MaximumTimeInterval`](api/time.md#light_curve.MaximumTimeInterval) | Maximum gap between consecutive observations | 1 |
| [`MinimumTimeInterval`](api/time.md#light_curve.MinimumTimeInterval) | Minimum gap between consecutive observations | 1 |
| [`ObservationCount`](api/time.md#light_curve.ObservationCount) | Number of observations | 1 |
| [`TimeMean`](api/time.md#light_curve.TimeMean) | Mean observation time | 1 |
| [`TimeStandardDeviation`](api/time.md#light_curve.TimeStandardDeviation) | Standard deviation of observation times | 1 |

### Periodogram

| Feature | Description | Outputs |
|---------|-------------|---------|
| [`Periodogram`](api/periodogram.md#light_curve.Periodogram) | Lomb–Scargle periodogram: period and power of the strongest peaks; supports `phase_features` to extract features from the phase-folded light curve at the best period | ≥2 |

### Non-linear parametric fits (transients, flux only)

| Feature | Description | Outputs |
|---------|-------------|---------|
| [`BazinFit`](api/fitting.md#light_curve.BazinFit) | Bazin function — 5-parameter rising/falling exponential fit for core-collapse SNe | 6 |
| [`LinexpFit`](api/fitting.md#light_curve.LinexpFit) | Linexp function — linear-times-exponential fit for core-collapse SNe | 5 |
| [`VillarFit`](api/fitting.md#light_curve.VillarFit) | Villar function — 7-parameter Gaussian+plateau fit for SN classification | 8 |

### Multiband parametric fit

| Feature | Description | Outputs |
|---------|-------------|---------|
| [`RainbowFit`](api/rainbow.md#light_curve.RainbowFit) | Multiband blackbody fit (Russeil+23) — bolometric flux + temperature evolution + SED spectral model | 4 to 10+ |

### Multiband

*Fast: all 4 features combined ~20 µs on 1,000 observations.*

| Feature | Description | Outputs |
|---------|-------------|---------|
| [`ColorOfMaximum`](api/multiband.md#light_curve.ColorOfMaximum) | Difference between maximum magnitudes of two bands | 1 |
| [`ColorOfMedian`](api/multiband.md#light_curve.ColorOfMedian) | Difference between median magnitudes of two bands | 1 |
| [`ColorOfMinimum`](api/multiband.md#light_curve.ColorOfMinimum) | Difference between minimum magnitudes of two bands | 1 |
| [`ColorSpread`](api/multiband.md#light_curve.ColorSpread) | Population std dev of per-band weighted mean magnitudes | 1 |

### Detection-based (experimental)

| Feature | Description | Outputs |
|---------|-------------|---------|
| [`FluxNNotDetBeforeFd`](api/detection.md#light_curve.FluxNNotDetBeforeFd) | Number of non-detections (flux) before the first detection | 1 |
| [`MagnitudeNNotDetBeforeFd`](api/detection.md#light_curve.MagnitudeNNotDetBeforeFd) | Number of non-detections (magnitude) before the first detection | 1 |

### Meta

| Feature | Description |
|---------|-------------|
| [`Extractor`](api/meta.md#light_curve.Extractor) | Combine multiple feature extractors into a single callable |
| [`Bins`](api/meta.md#light_curve.Bins) | Bin a time series, then apply any set of features to each bin |
