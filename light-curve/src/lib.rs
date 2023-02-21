use dmdt::DmDt;
use features as f;
use ln_prior::*;
use pyo3::prelude::*;

#[macro_use]
mod np_array;
mod check;
mod cont_array;
mod dmdt;
mod errors;
mod features;
mod ln_prior;

/// High-performance time-series feature extractor
///
/// The module provides a collection of features to be extracted from unevenly separated
/// time-series. This module if based on Rust crates `light-curve-feature` & `light-curve-dmdt`.
///
/// dm-lg(dt) maps generator is represented by `DmDt` class, while all other classes are
/// feature extractors
#[cfg_attr(
    feature = "fftw-static",
    deprecated(note = "fftw-static feature is deprecated")
)]
#[cfg_attr(
    feature = "fftw-dynamic",
    deprecated(note = "fftw-dynamic feature is deprecated")
)]
#[cfg_attr(feature = "mkl", deprecated(note = "mkl feature is deprecated"))]
#[pymodule]
fn light_curve(py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add("_built_with_gsl", cfg!(feature = "gsl"))?;
    m.add("_fft_backend", {
        #[cfg(feature = "fftw-mkl")]
        {
            "Intel MKL linked statically for FFTW"
        }
        #[cfg(all(not(feature = "fftw-mkl"), feature = "fftw-system"))]
        {
            "FFTW linked from system, may or may not be bundled into the package"
        }
        #[cfg(all(
            not(feature = "fftw-mkl"),
            not(feature = "fftw-system"),
            feature = "fftw-source"
        ))]
        {
            "FFTW built from source by fftw-src crate and statically linked into the module"
        }
        #[cfg(not(any(feature = "fftw-mkl", feature = "fftw-system", feature = "fftw-source")))]
        {
            compile_error!("One of fftw-mkl, fftw-system or fftw-source features is required");
        }
    })?;

    m.add_class::<DmDt>()?;

    m.add_class::<f::PyFeatureEvaluator>()?;

    m.add_class::<f::Extractor>()?;

    m.add_class::<f::Amplitude>()?;
    m.add_class::<f::AndersonDarlingNormal>()?;
    m.add_class::<f::BazinFit>()?;
    m.add_class::<f::BeyondNStd>()?;
    m.add_class::<f::Bins>()?;
    m.add_class::<f::Cusum>()?;
    m.add_class::<f::Duration>()?;
    m.add_class::<f::Eta>()?;
    m.add_class::<f::EtaE>()?;
    m.add_class::<f::ExcessVariance>()?;
    m.add_class::<f::InterPercentileRange>()?;
    m.add_class::<f::Kurtosis>()?;
    m.add_class::<f::LinearFit>()?;
    m.add_class::<f::LinearTrend>()?;
    m.add_class::<f::ObservationCount>()?;
    m.add_class::<f::OtsuSplit>()?;
    m.add_class::<f::MagnitudePercentageRatio>()?;
    m.add_class::<f::MaximumSlope>()?;
    m.add_class::<f::MaximumTimeInterval>()?;
    m.add_class::<f::Mean>()?;
    m.add_class::<f::MeanVariance>()?;
    m.add_class::<f::Median>()?;
    m.add_class::<f::MedianAbsoluteDeviation>()?;
    m.add_class::<f::MedianBufferRangePercentage>()?;
    m.add_class::<f::MinimumTimeInterval>()?;
    m.add_class::<f::PercentAmplitude>()?;
    m.add_class::<f::PercentDifferenceMagnitudePercentile>()?;
    m.add_class::<f::Periodogram>()?;
    m.add_class::<f::ReducedChi2>()?;
    m.add_class::<f::Skew>()?;
    m.add_class::<f::StandardDeviation>()?;
    m.add_class::<f::StetsonK>()?;
    m.add_class::<f::TimeMean>()?;
    m.add_class::<f::TimeStandardDeviation>()?;
    m.add_class::<f::VillarFit>()?;
    m.add_class::<f::WeightedMean>()?;

    register_ln_prior_submodule(py, m)?;

    Ok(())
}
