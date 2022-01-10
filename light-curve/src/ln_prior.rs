/// Prior classes and constructors for *Fit feature evaluators
use light_curve_feature as lcf;
use pyo3::prelude::*;

/// Logarithm of prior for *Fit feature extractors
///
/// Construct instances of this class using stand-alone functions
#[pyclass(module = "light_curve.light_curve_ext.ln_prior")]
#[derive(Clone)]
pub struct LnPrior1D(pub lcf::LnPrior1D);

/// "None" prior, its logarithm is zero
///
/// Returns
/// -------
/// LnPrior1D
#[pyfunction(text_signature = "()", module = "light_curve.light_curve_ext.ln_prior")]
fn none() -> LnPrior1D {
    LnPrior1D(lcf::LnPrior1D::none())
}

/// Log-normal prior
///
/// Parameters
/// ----------
/// mu : float
/// sigma : float
///
/// Returns
/// -------
/// LnPrior1D
///
/// https://en.wikipedia.org/wiki/Log-normal_distribution
#[pyfunction(
    text_signature = "(mu, sigma)",
    module = "light_curve.light_curve_ext.ln_prior"
)]
fn log_normal(mu: f64, sigma: f64) -> LnPrior1D {
    LnPrior1D(lcf::LnPrior1D::log_normal(mu, sigma))
}

/// Log-uniform prior
///
/// Parameters
/// ----------
/// left : float
///     Left border of the distribution (value, not logarithm)
/// right : float
///     Right border of the distribution
///
/// Returns
/// -------
/// LnPrior1D
#[pyfunction(
    text_signature = "(left, right)",
    module = "light_curve.light_curve_ext.ln_prior"
)]
fn log_uniform(left: f64, right: f64) -> LnPrior1D {
    LnPrior1D(lcf::LnPrior1D::log_uniform(left, right))
}

/// Normal prior
///
/// Parameters
/// ----------
/// mu : float
/// sigma : float
///
/// Returns
/// -------
/// LnPrior1D
#[pyfunction(
    text_signature = "(mu, sigma)",
    module = "light_curve.light_curve_ext.ln_prior"
)]
fn normal(mu: f64, sigma: f64) -> LnPrior1D {
    LnPrior1D(lcf::LnPrior1D::normal(mu, sigma))
}

/// Uniform prior
///
/// Parameters
/// ----------
/// left : float
///     Left border of the distribution
/// right : float
///     Right border of the distribution
///
/// Returns
/// -------
/// LnPrior1D
#[pyfunction(
    text_signature = "(left, right)",
    module = "light_curve.light_curve_ext.ln_prior"
)]
fn uniform(left: f64, right: f64) -> LnPrior1D {
    LnPrior1D(lcf::LnPrior1D::uniform(left, right))
}

/// Prior as a mixed distribution
///
/// Parameters
/// ----------
/// mix : list of (float, LnPrior1D) tuples
///     A mixed distribution represented as a list of (weight, LnPrior1D), the
///     mixed logarithm of prior is
///     ln(sum(norm_weight_i * exp(ln_prior_i(x))))
///     where norm_weight_i = weight_i / sum(weight_j)
#[pyfunction(
    text_signature = "mix",
    module = "light_curve.light_curve_ext.ln_prior"
)]
fn mix(mix: Vec<(f64, LnPrior1D)>) -> LnPrior1D {
    let priors = mix
        .into_iter()
        .map(|(weight, py_ln_prior)| (weight, py_ln_prior.0))
        .collect();
    LnPrior1D(lcf::LnPrior1D::mix(priors))
}

pub fn register_ln_prior_submodule(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "ln_prior")?;
    m.add_class::<LnPrior1D>()?;
    m.add_function(wrap_pyfunction!(none, m)?)?;
    m.add_function(wrap_pyfunction!(log_normal, m)?)?;
    m.add_function(wrap_pyfunction!(log_uniform, m)?)?;
    m.add_function(wrap_pyfunction!(normal, m)?)?;
    m.add_function(wrap_pyfunction!(uniform, m)?)?;
    m.add_function(wrap_pyfunction!(mix, m)?)?;
    parent_module.add_submodule(m)?;
    Ok(())
}
