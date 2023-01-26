/// Prior classes and constructors for *Fit feature evaluators
use crate::errors::{Exception, Res};

use light_curve_feature as lcf;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::{Deserialize, Serialize};

/// Logarithm of prior for *Fit feature extractors
///
/// Construct instances of this class using stand-alone functions. The constructor of this class
/// always returns `none` variant (see `ln_prior.none()`).
#[pyclass(module = "light_curve.light_curve_ext.ln_prior")]
#[derive(Clone, Serialize, Deserialize)]
pub struct LnPrior1D(pub lcf::LnPrior1D);

#[pymethods]
impl LnPrior1D {
    #[new]
    #[args()]
    fn __new__() -> Self {
        Self(lcf::LnPrior1D::none())
    }

    /// Used by pickle.load / pickle.loads
    #[args(state)]
    fn __setstate__(&mut self, state: &PyBytes) -> Res<()> {
        *self = serde_pickle::from_slice(state.as_bytes(), serde_pickle::DeOptions::new())
            .map_err(|err| {
                Exception::UnpicklingError(format!(
                    r#"Error happened on the Rust side when deserializing LnPrior1D: "{err}""#
                ))
            })?;
        Ok(())
    }

    /// Used by pickle.dump / pickle.dumps
    #[args()]
    fn __getstate__<'py>(&self, py: Python<'py>) -> Res<&'py PyBytes> {
        let vec_bytes =
            serde_pickle::to_vec(&self, serde_pickle::SerOptions::new()).map_err(|err| {
                Exception::PicklingError(format!(
                    r#"Error happened on the Rust side when serializing LnPrior1D: "{err}""#
                ))
            })?;
        Ok(PyBytes::new(py, &vec_bytes))
    }

    /// Used by copy.copy
    #[args()]
    fn __copy__(&self) -> Self {
        self.clone()
    }

    /// Used by copy.deepcopy
    #[args(memo)]
    fn __deepcopy__(&self, _memo: &PyAny) -> Self {
        self.clone()
    }
}

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
#[pyfunction(text_signature = "(mu, sigma)", module = "light_curve.light_curve_ext")]
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
    module = "light_curve.light_curve_ext"
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
#[pyfunction(text_signature = "mix", module = "light_curve.light_curve_ext")]
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
