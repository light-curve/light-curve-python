# Feature extractors

`light_curve` provides 40+ hand-crafted feature extractors for astrophysical light curves.
All share a common interface: callable objects with `.names` and `.descriptions` attributes.

```python
import light_curve as licu
import numpy as np

t = np.array([0.0, 1.2, 3.5, 7.1, 9.0])
m = np.array([15.1, 15.3, 14.9, 15.0, 15.2])
err = np.array([0.05, 0.05, 0.05, 0.05, 0.05])

ext = licu.Extractor(licu.Amplitude(), licu.BeyondNStd(nstd=1), licu.LinearFit())
result = ext(t, m, err)
print(dict(zip(ext.names, result)))
```

Use [`Extractor`](api/meta.md#light_curve.Extractor) to combine multiple features into a single callable,
or call each class directly for a single feature.
The `.many()` method processes a list of light curves in batch with reduced Python–Rust overhead.

See the [API reference](api/statistical.md) for full signatures, parameters, and equations.

---

## Feature table

### Statistical

| Feature | Description | Outputs |
|---------|-------------|---------|
| [`Amplitude`](api/statistical.md#light_curve.Amplitude) | Half peak-to-peak amplitude | 1 |
| [`AndersonDarlingNormal`](api/statistical.md#light_curve.AndersonDarlingNormal) | Unbiased Anderson–Darling normality test statistic | 1 |
| [`BeyondNStd`](api/statistical.md#light_curve.BeyondNStd) | Fraction of observations beyond \(n\,\sigma_m\) from \(\langle m \rangle\) | 1 |
| [`Chi2Pvar`](api/statistical.md#light_curve.Chi2Pvar) | Probability of variability from the \(\chi^2\) test | 1 |
| [`Cusum`](api/statistical.md#light_curve.Cusum) | Range of cumulative sums | 1 |
| [`ExcessVariance`](api/statistical.md#light_curve.ExcessVariance) | Measure of intrinsic variability amplitude | 1 |
| [`InterPercentileRange`](api/statistical.md#light_curve.InterPercentileRange) | \(Q(1-p) - Q(p)\) inter-percentile range | 1 |
| [`Kurtosis`](api/statistical.md#light_curve.Kurtosis) | Excess kurtosis of magnitude | 1 |
| [`Mean`](api/statistical.md#light_curve.Mean) | Mean magnitude | 1 |
| [`MeanVariance`](api/statistical.md#light_curve.MeanVariance) | Standard deviation to mean ratio | 1 |
| [`Median`](api/statistical.md#light_curve.Median) | Median magnitude | 1 |
| [`MedianAbsoluteDeviation`](api/statistical.md#light_curve.MedianAbsoluteDeviation) | Median of \(\lvert m_i - \mathrm{median}(m) \rvert\) | 1 |
| [`MedianBufferRangePercentage`](api/statistical.md#light_curve.MedianBufferRangePercentage) | Fraction of points within \(q \times \mathrm{amplitude}\) of median | 1 |
| [`OtsuSplit`](api/statistical.md#light_curve.OtsuSplit) | Otsu thresholding: bimodality measure (subset means, std devs, fraction) | 4 |
| [`PercentAmplitude`](api/statistical.md#light_curve.PercentAmplitude) | Maximum deviation of magnitude from its median | 1 |
| [`PercentDifferenceMagnitudePercentile`](api/statistical.md#light_curve.PercentDifferenceMagnitudePercentile) | Ratio of inter-percentile range to median | 1 |
| [`ReducedChi2`](api/statistical.md#light_curve.ReducedChi2) | Reduced \(\chi^2\) of magnitude measurements | 1 |
| [`Roms`](api/statistical.md#light_curve.Roms) | Robust median statistic *(experimental)* | 1 |
| [`Skew`](api/statistical.md#light_curve.Skew) | Skewness \(G_1\) of magnitude | 1 |
| [`StandardDeviation`](api/statistical.md#light_curve.StandardDeviation) | Standard deviation \(\sigma_m\) of magnitude | 1 |
| [`StetsonK`](api/statistical.md#light_curve.StetsonK) | Stetson \(K\) light-curve shape coefficient | 1 |
| [`WeightedMean`](api/statistical.md#light_curve.WeightedMean) | Inverse-variance weighted mean magnitude | 1 |

### Variability & trend

| Feature | Description | Outputs |
|---------|-------------|---------|
| [`Eta`](api/variability.md#light_curve.Eta) | Von Neumann \(\eta\) | 1 |
| [`EtaE`](api/variability.md#light_curve.EtaE) | \(\eta^e\), adapted for unevenly sampled time series | 1 |
| [`LaflerKinmanStringLength`](api/variability.md#light_curve.LaflerKinmanStringLength) | Lafler–Kinman string-length smoothness statistic | 1 |
| [`LinearFit`](api/variability.md#light_curve.LinearFit) | Slope, its error, and reduced \(\chi^2\) of the weighted linear fit | 3 |
| [`LinearTrend`](api/variability.md#light_curve.LinearTrend) | Slope, its error, and noise level of the unweighted linear fit | 3 |
| [`MaximumSlope`](api/variability.md#light_curve.MaximumSlope) | Maximum slope between consecutive observations | 1 |

### Time sampling

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

### Parametric fits (transients, flux only)

| Feature | Description | Outputs |
|---------|-------------|---------|
| [`BazinFit`](api/fitting.md#light_curve.BazinFit) | Bazin function — 5-parameter rising/falling exponential fit for core-collapse SNe | 6 |
| [`LinexpFit`](api/fitting.md#light_curve.LinexpFit) | Linexp function — linear-times-exponential fit for core-collapse SNe | 5 |
| [`VillarFit`](api/fitting.md#light_curve.VillarFit) | Villar function — 7-parameter Gaussian+plateau fit for SN classification | 8 |

### Detection-based (experimental)

| Feature | Description | Outputs |
|---------|-------------|---------|
| [`FluxNNotDetBeforeFd`](api/detection.md#light_curve.FluxNNotDetBeforeFd) | Number of non-detections (flux) before the first detection | 1 |
| [`MagnitudeNNotDetBeforeFd`](api/detection.md#light_curve.MagnitudeNNotDetBeforeFd) | Number of non-detections (magnitude) before the first detection | 1 |

### Multiband (experimental)

| Feature | Description | Outputs |
|---------|-------------|---------|
| [`ColorOfMedian`](multiband/api.md#light_curve.ColorOfMedian) | Difference between median magnitudes of two bands | 1 |
| [`RainbowFit`](multiband/api.md#light_curve.RainbowFit) | Multiband blackbody fit (Russeil+23) — bolometric flux + temperature evolution | 8 |

### Meta

| Feature | Description |
|---------|-------------|
| [`Extractor`](api/meta.md#light_curve.Extractor) | Combine multiple feature extractors into a single callable |
| [`Bins`](api/meta.md#light_curve.Bins) | Bin a time series, then apply any set of features to each bin |
