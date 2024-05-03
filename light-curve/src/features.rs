use crate::check::{check_finite, check_no_nans, is_sorted};
use crate::cont_array::ContCowArray;
use crate::errors::{Exception, Res};
use crate::ln_prior::LnPrior1D;
use crate::np_array::Arr;
use crate::transform::{parse_transform, StockTransformer};

use const_format::formatcp;
use conv::ConvUtil;
use itertools::Itertools;
use light_curve_feature::{self as lcf, prelude::*, DataSample};
use macro_const::macro_const;
use ndarray::IntoNdProducer;
use numpy::prelude::*;
use numpy::{PyArray1, PyUntypedArray};
use once_cell::sync::OnceCell;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyTuple};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::TryInto;

// Details of pickle support implementation
// ----------------------------------------
// [PyFeatureEvaluator] implements __getstate__ and __setstate__ required for pickle serialisation,
// which gives the support of pickle protocols 2+. However it is not enough for child classes with
// mandatory constructor arguments since __setstate__(self, state) is a method applied after
// __new__ is called. Thus we implement __getnewargs__ (or __getnewargs_ex__ when constructor has
// keyword-only arguments) for such classes. Despite the "standard" way we return some default
// arguments from this method and de-facto re-create the underlying Rust objects during
// __setstate__, which could reduce performance of deserialising. We also make this method static
// to use it in tests, which is also a bit weird thing to do. We use pickle as a serialization
// format for it, so all Rust object internals can be inspected from Python.

type PyLcs<'py> = Vec<(
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Option<Bound<'py, PyAny>>,
)>;

const ATTRIBUTES_DOC: &str = r#"Attributes
----------
names : list of str
    Feature names
descriptions : list of str
    Feature descriptions"#;

const METHOD_CALL_DOC: &str = r#"__call__(self, t, m, sigma=None, *, sorted=None, check=True, fill_value=None)
    Extract features and return them as a numpy array

    Parameters
    ----------
    t : numpy.ndarray of np.float32 or np.float64 dtype
        Time moments
    m : numpy.ndarray of the same dtype and size as t
        Signal in magnitude or fluxes. Refer to the feature description to
        decide which would work better in your case
    sigma : numpy.ndarray of the same dtype and size as t, optional
        Observation error, if None it is assumed to be unity
    sorted : bool or None, optional
        Specifies if input array are sorted by time moments.
        True is for certainly sorted, False is for unsorted.
        If None is specified than sorting is checked and an exception is
        raised for unsorted `t`
    check : bool, optional
        Check all input arrays for NaNs, `t` and `m` for infinite values
    fill_value : float or None, optional
        Value to fill invalid feature values, for example if count of
        observations is not enough to find a proper value.
        None causes exception for invalid features

    Returns
    -------
    ndarray of np.float32 or np.float64
        Extracted feature array"#;

macro_const! {
    const METHOD_MANY_DOC: &str = r#"
many(self, lcs, *, sorted=None, check=True, fill_value=None, n_jobs=-1)
    Parallel light curve feature extraction

    It is a parallel executed equivalent of
    >>> def many(self, lcs, *, sorted=None, check=True, fill_value=None):
    ...     return np.stack(
    ...         [
    ...             self(
    ...                 *lc,
    ...                 sorted=sorted,
    ...                 check=check,
    ...                 fill_value=fill_value
    ...             )
    ...             for lc in lcs
    ...         ]
    ...     )

    Parameters
    ----------
    lcs : list ot (t, m, sigma)
        A collection of light curves packed into three-tuples, all light curves
        must be represented by numpy.ndarray of the same dtype. See __call__
        documentation for details
    sorted : bool or None, optional
        Specifies if input array are sorted by time moments, see __call__
        documentation for details
    check : bool, optional
        Check all input arrays for NaNs, `t` and `m` for infinite values
    fill_value : float or None, optional
        Fill invalid values by this or raise an exception if None
    n_jobs : int
        Number of tasks to run in paralell. Default is -1 which means run as
        many jobs as CPU count. See rayon rust crate documentation for
        details"#;
}

const METHODS_DOC: &str = formatcp!(
    r#"Methods
-------
{}
{}"#,
    METHOD_CALL_DOC,
    METHOD_MANY_DOC,
);

const COMMON_FEATURE_DOC: &str = formatcp!("\n{}\n\n{}\n", ATTRIBUTES_DOC, METHODS_DOC);

fn transform_parameter_doc(default: StockTransformer) -> String {
    let default_name: &str = default.into();
    let variants = StockTransformer::all_variants().format_with("\n     - ", |variant, fmt| {
        let name: &str = variant.into();
        let doc = variant.doc().trim();
        fmt(&format_args!("'{name}' - {doc}"))
    });
    format!(
        r#"transform : str or bool or None
    Transformer to apply to the feature values. If str, must be one of:
     - 'default' - use default transformer for the feature, it same as giving
       True. The default for this feature is '{default_name}'
     - {variants}
    If bool, must be True to use default transformer or False to disable.
    If None, no transformation is applied"#,
        default_name = default_name,
        variants = variants,
    )
}

type PyLightCurve<'a, T> = (Arr<'a, T>, Arr<'a, T>, Option<Arr<'a, T>>);

#[derive(Serialize, Deserialize, Clone)]
#[pyclass(
    subclass,
    name = "_FeatureEvaluator",
    module = "light_curve.light_curve_ext"
)]
pub struct PyFeatureEvaluator {
    feature_evaluator_f32: lcf::Feature<f32>,
    feature_evaluator_f64: lcf::Feature<f64>,
}

impl PyFeatureEvaluator {
    fn with_transform(
        (fe_f32, fe_f64): (lcf::Feature<f32>, lcf::Feature<f64>),
        (tr_f32, tr_f64): (lcf::Transformer<f32>, lcf::Transformer<f64>),
    ) -> Res<Self> {
        Ok(Self {
            feature_evaluator_f32: lcf::Transformed::new(fe_f32, tr_f32)
                .map_err(|err| {
                    Exception::ValueError(format!(
                        "feature and transformation are incompatible: {:?}",
                        err
                    ))
                })?
                .into(),
            feature_evaluator_f64: lcf::Transformed::new(fe_f64, tr_f64)
                .map_err(|err| {
                    Exception::ValueError(format!(
                        "feature and transformation are incompatible: {:?}",
                        err
                    ))
                })?
                .into(),
        })
    }

    fn with_py_transform(
        fe_f32: lcf::Feature<f32>,
        fe_f64: lcf::Feature<f64>,
        transform: Option<Bound<PyAny>>,
        default_transformer: StockTransformer,
    ) -> Res<Self> {
        let transform = parse_transform(transform, default_transformer)?;
        match transform {
            Some(transform) => Self::with_transform((fe_f32, fe_f64), transform.into()),
            None => Ok(Self {
                feature_evaluator_f32: fe_f32,
                feature_evaluator_f64: fe_f64,
            }),
        }
    }

    fn ts_from_numpy<'a, T>(
        feature_evaluator: &lcf::Feature<T>,
        t: &'a Arr<'a, T>,
        m: &'a Arr<'a, T>,
        sigma: &'a Option<Arr<'a, T>>,
        sorted: Option<bool>,
        check: bool,
        is_t_required: bool,
    ) -> Res<lcf::TimeSeries<'a, T>>
    where
        T: lcf::Float + numpy::Element,
    {
        if t.len() != m.len() {
            return Err(Exception::ValueError(
                "t and m must have the same size".to_string(),
            ));
        }
        if let Some(ref sigma) = sigma {
            if t.len() != sigma.len() {
                return Err(Exception::ValueError(
                    "t and sigma must have the same size".to_string(),
                ));
            }
        }

        let mut t: lcf::DataSample<_> = if is_t_required || t.is_contiguous() {
            let t = t.as_array();
            if check {
                check_finite(t)?;
            }
            t.into()
        } else {
            T::array0_unity().broadcast(t.len()).unwrap().into()
        };
        match sorted {
            Some(true) => {}
            Some(false) => {
                return Err(Exception::NotImplementedError(
                    "sorting is not implemented, please provide time-sorted arrays".to_string(),
                ))
            }
            None => {
                if feature_evaluator.is_sorting_required() & !is_sorted(t.as_slice()) {
                    return Err(Exception::ValueError(
                        "t must be in ascending order".to_string(),
                    ));
                }
            }
        }

        let m: lcf::DataSample<_> = if feature_evaluator.is_m_required() || m.is_contiguous() {
            let m = m.as_array();
            if check {
                check_finite(m)?;
            }
            m.into()
        } else {
            T::array0_unity().broadcast(m.len()).unwrap().into()
        };

        let w = match sigma.as_ref() {
            Some(sigma) => {
                if feature_evaluator.is_w_required() {
                    let sigma = sigma.as_array();
                    if check {
                        check_no_nans(sigma)?;
                    }
                    let mut a = sigma.to_owned();
                    a.mapv_inplace(|x| x.powi(-2));
                    Some(a)
                } else {
                    None
                }
            }
            None => None,
        };

        let ts = match w {
            Some(w) => lcf::TimeSeries::new(t, m, w),
            None => lcf::TimeSeries::new_without_weight(t, m),
        };

        Ok(ts)
    }

    #[allow(clippy::too_many_arguments)]
    fn call_impl<'py, T>(
        feature_evaluator: &lcf::Feature<T>,
        py: Python<'py>,
        t: Arr<'py, T>,
        m: Arr<'py, T>,
        sigma: Option<Arr<'py, T>>,
        sorted: Option<bool>,
        check: bool,
        is_t_required: bool,
        fill_value: Option<T>,
    ) -> Res<Bound<'py, PyUntypedArray>>
    where
        T: lcf::Float + numpy::Element,
    {
        let mut ts = Self::ts_from_numpy(
            feature_evaluator,
            &t,
            &m,
            &sigma,
            sorted,
            check,
            is_t_required,
        )?;

        let result = match fill_value {
            Some(x) => feature_evaluator.eval_or_fill(&mut ts, x),
            None => feature_evaluator
                .eval(&mut ts)
                .map_err(|e| Exception::ValueError(e.to_string()))?,
        };
        let array = PyArray1::from_vec_bound(py, result);
        Ok(array.as_untyped().clone())
    }

    #[allow(clippy::too_many_arguments)]
    fn py_many<'py, T>(
        &self,
        feature_evaluator: &lcf::Feature<T>,
        py: Python<'py>,
        lcs: PyLcs<'py>,
        sorted: Option<bool>,
        check: bool,
        fill_value: Option<T>,
        n_jobs: i64,
    ) -> Res<Bound<'py, PyUntypedArray>>
    where
        T: lcf::Float + numpy::Element,
    {
        let wrapped_lcs = lcs
            .into_iter()
            .enumerate()
            .map(|(i, (t, m, sigma))| {
                let t = t.downcast::<PyArray1<T>>().map(|a| a.readonly());
                let m = m.downcast::<PyArray1<T>>().map(|a| a.readonly());
                let sigma = match &sigma {
                    Some(sigma) => sigma.downcast::<PyArray1<T>>().map(|a| Some(a.readonly())),
                    None => Ok(None),
                };

                match (t, m, sigma) {
                    (Ok(t), Ok(m), Ok(sigma)) => Ok((t, m, sigma)),
                    _ => Err(Exception::TypeError(format!(
                        "lcs[{}] elements have mismatched dtype with the lc[0][0] which is {}",
                        i,
                        std::any::type_name::<T>()
                    ))),
                }
            })
            .collect::<Res<Vec<_>>>()?;
        Ok(Self::many_impl(
            feature_evaluator,
            wrapped_lcs,
            sorted,
            check,
            self.is_t_required(sorted),
            fill_value,
            n_jobs,
        )?
        .into_pyarray_bound(py)
        .as_untyped()
        .clone())
    }

    fn many_impl<T>(
        feature_evaluator: &lcf::Feature<T>,
        lcs: Vec<PyLightCurve<T>>,
        sorted: Option<bool>,
        check: bool,
        is_t_required: bool,
        fill_value: Option<T>,
        n_jobs: i64,
    ) -> Res<ndarray::Array2<T>>
    where
        T: lcf::Float + numpy::Element,
    {
        let n_jobs = if n_jobs < 0 { 0 } else { n_jobs as usize };

        let mut result = ndarray::Array2::zeros((lcs.len(), feature_evaluator.size_hint()));

        let mut tss = lcs
            .iter()
            .map(|(t, m, sigma)| {
                Self::ts_from_numpy(feature_evaluator, t, m, sigma, sorted, check, is_t_required)
            })
            .collect::<Result<Vec<_>, _>>()?;

        rayon::ThreadPoolBuilder::new()
            .num_threads(n_jobs)
            .build()
            .unwrap()
            .install(|| {
                ndarray::Zip::from(result.outer_iter_mut())
                    .and((&mut tss).into_producer())
                    .into_par_iter()
                    .try_for_each::<_, Res<_>>(|(mut map, ts)| {
                        let features: ndarray::Array1<_> = match fill_value {
                            Some(x) => feature_evaluator.eval_or_fill(ts, x),
                            None => feature_evaluator
                                .eval(ts)
                                .map_err(|e| Exception::ValueError(e.to_string()))?,
                        }
                        .into();
                        map.assign(&features);
                        Ok(())
                    })
            })?;
        Ok(result)
    }

    fn is_t_required(&self, sorted: Option<bool>) -> bool {
        match (
            self.feature_evaluator_f64.is_t_required(),
            self.feature_evaluator_f64.is_sorting_required(),
            sorted,
        ) {
            // feature requires t
            (true, _, _) => true,
            // t is required because sorting is required and data can be unsorted
            (false, true, Some(false)) | (false, true, None) => true,
            // sorting is required but user guarantees that data is already sorted
            (false, true, Some(true)) => false,
            // neither t or sorting is required
            (false, false, _) => false,
        }
    }
}

#[pymethods]
impl PyFeatureEvaluator {
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        t,
        m,
        sigma = None,
        *,
        sorted = None,
        check = true,
        fill_value = None
    ))]
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        t: Bound<'py, PyAny>,
        m: Bound<'py, PyAny>,
        sigma: Option<Bound<'py, PyAny>>,
        sorted: Option<bool>,
        check: bool,
        fill_value: Option<f64>,
    ) -> Res<Bound<'py, PyUntypedArray>> {
        if let Some(sigma) = sigma {
            dtype_dispatch!(
                |t, m, sigma| {
                    Self::call_impl(
                        &self.feature_evaluator_f32,
                        py,
                        t,
                        m,
                        Some(sigma),
                        sorted,
                        check,
                        self.is_t_required(sorted),
                        fill_value.map(|v| v as f32),
                    )
                },
                |t, m, sigma| {
                    Self::call_impl(
                        &self.feature_evaluator_f64,
                        py,
                        t,
                        m,
                        Some(sigma),
                        sorted,
                        check,
                        self.is_t_required(sorted),
                        fill_value,
                    )
                },
                t,
                =m,
                =sigma
            )
        } else {
            dtype_dispatch!(
                |t, m| {
                    Self::call_impl(
                        &self.feature_evaluator_f32,
                        py,
                        t,
                        m,
                        None,
                        sorted,
                        check,
                        self.is_t_required(sorted),
                        fill_value.map(|v| v as f32),
                    )
                },
                |t, m| {
                    Self::call_impl(
                        &self.feature_evaluator_f64,
                        py,
                        t,
                        m,
                        None,
                        sorted,
                        check,
                        self.is_t_required(sorted),
                        fill_value,
                    )
                },
                t,
                =m,
            )
        }
    }

    #[doc = METHOD_MANY_DOC!()]
    #[pyo3(signature = (lcs, *, sorted=None, check=true, fill_value=None, n_jobs=-1))]
    fn many<'py>(
        &self,
        py: Python<'py>,
        lcs: PyLcs<'py>,
        sorted: Option<bool>,
        check: bool,
        fill_value: Option<f64>,
        n_jobs: i64,
    ) -> Res<Bound<'py, PyUntypedArray>> {
        if lcs.is_empty() {
            Err(Exception::ValueError("lcs is empty".to_string()))
        } else {
            dtype_dispatch!(
                |_first_t| {
                    self.py_many(
                        &self.feature_evaluator_f32,
                        py,
                        lcs,
                        sorted,
                        check,
                        fill_value.map(|v| v as f32),
                        n_jobs,
                    )
                },
                |_first_t| {
                    self.py_many(
                        &self.feature_evaluator_f64,
                        py,
                        lcs,
                        sorted,
                        check,
                        fill_value,
                        n_jobs,
                    )
                },
                lcs[0].0
            )
        }
    }

    /// Serialize feature evaluator to json string
    fn to_json(&self) -> String {
        serde_json::to_string(&self.feature_evaluator_f64).unwrap()
    }

    /// Feature names
    #[getter]
    fn names(&self) -> Vec<&str> {
        self.feature_evaluator_f64.get_names()
    }

    /// Feature descriptions
    #[getter]
    fn descriptions(&self) -> Vec<&str> {
        self.feature_evaluator_f64.get_descriptions()
    }

    /// Used by pickle.load / pickle.loads
    fn __setstate__(&mut self, state: Bound<PyBytes>) -> Res<()> {
        *self = serde_pickle::from_slice(state.as_bytes(), serde_pickle::DeOptions::new())
            .map_err(|err| {
                Exception::UnpicklingError(format!(
                    r#"Error happened on the Rust side when deserializing _FeatureEvaluator: "{err}""#
                ))
            })?;
        Ok(())
    }

    /// Used by pickle.dump / pickle.dumps
    fn __getstate__<'py>(&self, py: Python<'py>) -> Res<Bound<'py, PyBytes>> {
        let vec_bytes =
            serde_pickle::to_vec(&self, serde_pickle::SerOptions::new()).map_err(|err| {
                Exception::PicklingError(format!(
                    r#"Error happened on the Rust side when serializing _FeatureEvaluator: "{err}""#
                ))
            })?;
        Ok(PyBytes::new_bound(py, &vec_bytes))
    }

    /// Used by copy.copy
    fn __copy__(&self) -> Self {
        self.clone()
    }

    /// Used by copy.deepcopy
    fn __deepcopy__(&self, _memo: Bound<PyAny>) -> Self {
        self.clone()
    }
}

#[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
pub struct Extractor {}

#[pymethods]
impl Extractor {
    #[new]
    #[pyo3(signature = (*features, transform=None))]
    fn __new__(
        features: Bound<PyTuple>,
        transform: Option<Bound<PyAny>>,
    ) -> PyResult<(Self, PyFeatureEvaluator)> {
        if transform.is_some() {
            return Err(Exception::NotImplementedError(
                "transform is not implemented for Extractor, transform individual features instead"
                    .to_string(),
            )
            .into());
        }
        let evals_iter = features.iter_borrowed().map(|arg| {
            arg.extract::<PyFeatureEvaluator>().map(|fe| {
                (
                    fe.feature_evaluator_f32.clone(),
                    fe.feature_evaluator_f64.clone(),
                )
            })
        });
        let (evals_f32, evals_f64) =
            itertools::process_results(evals_iter, |iter| iter.unzip::<_, _, Vec<_>, Vec<_>>())?;
        Ok((
            Self {},
            PyFeatureEvaluator {
                feature_evaluator_f32: lcf::FeatureExtractor::new(evals_f32).into(),
                feature_evaluator_f64: lcf::FeatureExtractor::new(evals_f64).into(),
            },
        ))
    }

    #[classattr]
    fn __doc__() -> String {
        format!(
            r#"{}

Parameters
----------
*features : iterable
    Feature objects
transform : None, optional
    Not implemented for Extractor, transform individual features instead
{}
"#,
            lcf::FeatureExtractor::<f64, lcf::Feature<f64>>::doc().trim_start(),
            COMMON_FEATURE_DOC,
        )
    }
}

macro_rules! impl_stock_transform {
    ($name: ident, $default_transform: expr $(,)?) => {
        impl $name {
            const DEFAULT_TRANSFORMER: StockTransformer = $default_transform;
        }

        #[pymethods]
        impl $name {
            /// Supported transform names
            #[classattr]
            fn supported_transforms() -> Vec<&'static str> {
                StockTransformer::all_names().collect()
            }

            /// Default transform name
            #[classattr]
            fn default_transform() -> &'static str {
                Self::DEFAULT_TRANSFORMER.into()
            }
        }
    };
}

macro_rules! evaluator {
    ($name: ident, $eval: ty, $default_transform: expr $(,)?) => {
        #[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
        pub struct $name {}

        impl_stock_transform!($name, $default_transform);

        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature=(*, transform=None))]
            fn __new__(transform: Option<Bound<PyAny>>) -> Res<PyClassInitializer<Self>> {
                let base = PyFeatureEvaluator::with_py_transform(
                    <$eval>::new().into(),
                    <$eval>::new().into(),
                    transform,
                    Self::DEFAULT_TRANSFORMER,
                )?;
                Ok(PyClassInitializer::from(base).add_subclass(Self {}))
            }

            #[classattr]
            fn __doc__() -> String {
                format!(
                    r#"{header}
Parameters
----------
{transform_variant}
{footer}"#,
                    header = <$eval>::doc().trim_start(),
                    transform_variant = transform_parameter_doc(Self::DEFAULT_TRANSFORMER),
                    footer = COMMON_FEATURE_DOC
                )
            }
        }
    };
}

const N_ALGO_CURVE_FIT_CERES: usize = {
    #[cfg(any(feature = "ceres-source", feature = "ceres-system"))]
    {
        2
    }
    #[cfg(not(any(feature = "ceres-source", feature = "ceres-system")))]
    {
        0
    }
};
const N_ALGO_CURVE_FIT_GSL: usize = {
    #[cfg(feature = "gsl")]
    {
        2
    }
    #[cfg(not(feature = "gsl"))]
    {
        0
    }
};
const N_ALGO_CURVE_FIT_PURE_MCMC: usize = 1;
const N_ALGO_CURVE_FIT: usize =
    N_ALGO_CURVE_FIT_CERES + N_ALGO_CURVE_FIT_GSL + N_ALGO_CURVE_FIT_PURE_MCMC;

const SUPPORTED_ALGORITHMS_CURVE_FIT: [&str; N_ALGO_CURVE_FIT] = [
    "mcmc",
    #[cfg(any(feature = "ceres-source", feature = "ceres-system"))]
    "ceres",
    #[cfg(any(feature = "ceres-source", feature = "ceres-system"))]
    "mcmc-ceres",
    #[cfg(feature = "gsl")]
    "lmsder",
    #[cfg(feature = "gsl")]
    "mcmc-lmsder",
];

macro_const! {
    const FIT_METHOD_MODEL_DOC: &str = r#"model(t, params)
    Underlying parametric model function

    Parameters
    ----------
    t : np.ndarray of np.float32 or np.float64
        Time moments, can be unsorted
    params : np.ndaarray of np.float32 or np.float64
        Parameters of the model, this array can be longer than actual parameter
        list, the beginning part of the array will be used in this case, see
        Examples section in the class documentation.

    Returns
    -------
    np.ndarray of np.float32 or np.float64
        Array of model values corresponded to the given time moments
"#;
}

#[derive(FromPyObject)]
pub(crate) enum FitLnPrior<'a> {
    #[pyo3(transparent, annotation = "str")]
    Name(&'a str),
    #[pyo3(transparent, annotation = "list[LnPrior]")]
    ListLnPrior1D(Vec<LnPrior1D>),
}

macro_rules! fit_evaluator {
    ($name: ident, $eval: ty, $ib: ty, $transform: expr, $nparam: literal, $ln_prior_by_str: tt, $ln_prior_doc: literal $(,)?) => {
        #[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
        pub struct $name {}

        impl $name {
            fn supported_algorithms_str() -> String {
                return SUPPORTED_ALGORITHMS_CURVE_FIT.join(", ");
            }

            fn lazy_default() -> &'static $eval {
                static DEFAULT: OnceCell<$eval> = OnceCell::new();
                DEFAULT.get_or_init(|| <$eval>::default())
            }

            fn lazy_names() -> &'static Vec<&'static str> {
                static NAMES: OnceCell<Vec<&str>> = OnceCell::new();
                NAMES.get_or_init(|| Self::lazy_default().get_names())
            }

            fn lazy_descriptions() -> &'static Vec<&'static str> {
                static DESC: OnceCell<Vec<&str>> = OnceCell::new();
                DESC.get_or_init(|| Self::lazy_default().get_descriptions())
            }
        }

        impl $name {
            fn model_impl<T>(t: Arr<T>, params: Arr<T>) -> ndarray::Array1<T>
            where
                T: lcf::Float + numpy::Element,
            {
                let params = ContCowArray::from_view(params.as_array(), true);
                t.as_array().mapv(|x| <$eval>::f(x, params.as_slice()))
            }

            fn default_lmsder_iterations() -> Option<u16> {
                #[cfg(feature = "gsl")]
                {
                    lcf::LmsderCurveFit::default_niterations().into()
                }
                #[cfg(not(feature = "gsl"))]
                {
                    None
                }
            }

            fn default_ceres_iterations() -> Option<u16> {
                #[cfg(any(feature = "ceres-source", feature = "ceres-system"))]
                {
                    lcf::CeresCurveFit::default_niterations().into()
                }
                #[cfg(not(any(feature = "ceres-source", feature = "ceres-system")))]
                {
                    None
                }
            }
        }

        #[allow(clippy::too_many_arguments)]
        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (
                algorithm,
                *,
                mcmc_niter = lcf::McmcCurveFit::default_niterations(),
                lmsder_niter = Self::default_lmsder_iterations(),
                ceres_niter = Self::default_ceres_iterations(),
                ceres_loss_reg = None,
                init = None,
                bounds = None,
                ln_prior = None,
                transform = None,
            ))]
            fn __new__(
                algorithm: &str,
                mcmc_niter: Option<u32>,
                // The first Option is for Python's None, the second is for compile-time None from
                // Self::default_lmsder_iterations()
                lmsder_niter: Option<Option<u16>>,
                // The first Option is for Python's None, the second is for compile-time None from
                // Self::default_ceres_iterations()
                ceres_niter: Option<Option<u16>>,
                ceres_loss_reg: Option<f64>,
                init: Option<Vec<Option<f64>>>,
                bounds: Option<Vec<(Option<f64>, Option<f64>)>>,
                ln_prior: Option<FitLnPrior<'_>>,
                transform: Option<Bound<PyAny>>,
            ) -> PyResult<(Self, PyFeatureEvaluator)> {
                let mcmc_niter = mcmc_niter.unwrap_or_else(lcf::McmcCurveFit::default_niterations);

                #[cfg(feature = "gsl")]
                let lmsder_fit: lcf::CurveFitAlgorithm = lcf::LmsderCurveFit::new(
                    lmsder_niter.unwrap_or_else(Self::default_lmsder_iterations).expect("logical error: default lmsder_niter is None but GSL is enabled"),
                )
                .into();
                #[cfg(not(feature = "gsl"))]
                if lmsder_niter.flatten().is_some() {
                    return Err(PyValueError::new_err(
                        "Compiled without GSL support, lmsder_niter is not supported",
                    ));
                }

                #[cfg(any(feature = "ceres-source", feature = "ceres-system"))]
                let ceres_fit: lcf::CurveFitAlgorithm = lcf::CeresCurveFit::new(
                    ceres_niter.unwrap_or_else(Self::default_ceres_iterations).expect("logical error: default ceres_niter is None but Ceres is enabled"),
                    ceres_loss_reg.or_else(lcf::CeresCurveFit::default_loss_factor),
                ).into();
                #[cfg(not(any(feature = "ceres-source", feature = "ceres-system")))]
                if ceres_niter.flatten().is_some() || ceres_loss_reg.is_some() {
                    return Err(PyValueError::new_err(
                        "Compiled without Ceres support, ceres_niter and ceres_loss_reg are not supported",
                    ));
                }

                let init_bounds = match (init, bounds) {
                    (Some(init), Some(bounds)) => Some((init, bounds)),
                    (Some(init), None) => {
                        let size = init.len();
                        Some((init, (0..size).map(|_| (None, None)).collect()))
                    }
                    (None, Some(bounds)) => Some((bounds.iter().map(|_| None).collect(), bounds)),
                    (None, None) => None,
                };
                let init_bounds = match init_bounds {
                    Some((init, bounds)) => {
                        let (lower, upper): (Vec<_>, Vec<_>) = bounds.into_iter().unzip();
                        <$ib>::option_arrays(
                            init.try_into().map_err(|_| {
                                Exception::ValueError("init has a wrong size".into())
                            })?,
                            lower.try_into().map_err(|_| {
                                Exception::ValueError("bounds has a wrong size".into())
                            })?,
                            upper.try_into().map_err(|_| {
                                Exception::ValueError("bounds has a wrong size".into())
                            })?,
                        )
                    }
                    None => <$ib>::default(),
                };

                let ln_prior = match ln_prior {
                    Some(ln_prior) => match ln_prior {
                        FitLnPrior::Name(s) => match s $ln_prior_by_str,
                        FitLnPrior::ListLnPrior1D(v) => {
                            let v: Vec<_> = v.into_iter().map(|py_ln_prior1d| py_ln_prior1d.0).collect();
                            lcf::LnPrior::ind_components(
                                v.try_into().map_err(|v: Vec<_>| Exception::ValueError(
                                    format!(
                                        "ln_prior must have length of {}, not {}",
                                        $nparam,
                                        v.len())
                                    )
                                )?
                            ).into()
                        }
                    },
                    None => lcf::LnPrior::none().into(),
                };

                let curve_fit_algorithm: lcf::CurveFitAlgorithm = match algorithm {
                    "mcmc" => lcf::McmcCurveFit::new(mcmc_niter, None).into(),
                    #[cfg(any(feature = "ceres-source", feature = "ceres-system"))]
                    "ceres" => ceres_fit,
                    #[cfg(any(feature = "ceres-source", feature = "ceres-system"))]
                    "mcmc-ceres" => lcf::McmcCurveFit::new(mcmc_niter, Some(ceres_fit)).into(),
                    #[cfg(feature = "gsl")]
                    "lmsder" => lmsder_fit,
                    #[cfg(feature = "gsl")]
                    "mcmc-lmsder" => lcf::McmcCurveFit::new(mcmc_niter, Some(lmsder_fit)).into(),
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            r#"wrong algorithm value "{}", supported values are: {}"#,
                            algorithm,
                            Self::supported_algorithms_str()
                        )))
                    }
                };

                let (fe_f32, fe_f64) = (<$eval>::new(
                            curve_fit_algorithm.clone(),
                            ln_prior.clone(),
                            init_bounds.clone(),
                        )
                        .into(),
                <$eval>::new(
                            curve_fit_algorithm,
                            ln_prior,
                            init_bounds,
                        )
                        .into(),);

                let make_transformation = match transform {
                    None => false,
                    Some(py_transform) => match py_transform.downcast::<PyBool>() {
                        Ok(py_bool) => py_bool.is_true(),
                        Err(_) => return Err(PyValueError::new_err(
                            "transform must be a bool or None, other types are not implemented yet",
                        )),
                    }
                };
                let fe = if make_transformation {
                    PyFeatureEvaluator::with_transform((fe_f32, fe_f64), ($transform.into(), $transform.into()))?
                } else {
                    PyFeatureEvaluator {
                        feature_evaluator_f32: fe_f32,
                        feature_evaluator_f64: fe_f64,
                    }
                };

                Ok((Self{}, fe))
            }

            /// Required by pickle.dump / pickle.dumps
            #[staticmethod]
            fn __getnewargs__() -> (&'static str,) {
                ("mcmc",)
            }

            #[doc = FIT_METHOD_MODEL_DOC!()]
            #[staticmethod]
            fn model<'py>(
                py: Python<'py>,
                t: Bound<'py, PyAny>,
                params: Bound<'py, PyAny>,
            ) -> Res<Bound<'py, PyUntypedArray>> {
                dtype_dispatch!({
                    |t, params| Ok(Self::model_impl(t, params).into_pyarray_bound(py).as_untyped().clone())
                }(t, !=params))
            }

            #[classattr]
            fn supported_algorithms() -> [&'static str; N_ALGO_CURVE_FIT] {
                return SUPPORTED_ALGORITHMS_CURVE_FIT;
            }

            #[classattr]
            fn __doc__() -> String {
                #[cfg(any(feature = "ceres-source", feature = "ceres-system"))]
                let ceres_args = format!(
                    r#"ceres_niter : int, optional
    Number of Ceres iterations, default is {niter}
ceres_loss_reg : float, optional
    Ceres loss regularization, default is to use square norm as is, if set to
    a number, the loss function is reqgualized to descriminate outlier
    residuals larger than this value.
    Default is None which means no regularization.
"#,
                    niter = lcf::CeresCurveFit::default_niterations()
                );
                #[cfg(not(any(feature = "ceres-source", feature = "ceres-system")))]
                let ceres_args = "";
                #[cfg(feature = "gsl")]
                let lmsder_niter = format!(
                    r#"lmsder_niter : int, optional
    Number of LMSDER iterations, default is {}
"#,
                    lcf::LmsderCurveFit::default_niterations()
                );
                #[cfg(not(feature = "gsl"))]
                let lmsder_niter = "";

                let names_descriptions: String = Self::lazy_names().iter().zip(Self::lazy_descriptions()).map(|(name, description)| {
                    format!(" - {}: {}\n", name, description)
                }).collect();

                format!(
                    r#"{intro}
Names and description of the output features:
{names_descriptions}
Parameters
----------
algorithm : str
    Non-linear least-square algorithm, supported values are:
    {supported_algo}.
mcmc_niter : int, optional
    Number of MCMC iterations, default is {mcmc_niter}
{ceres_args}{lmsder_niter}init : list or None, optional
    Initial conditions, must be `None` or a `list` of `float`s or `None`s.
    The length of the list must be {nparam}, `None` values will be replaced
    with some defauls values. It is supported by MCMC only
bounds : list of tuples or None, optional
    Boundary conditions, must be `None` or a `list` of `tuple`s of `float`s or
    `None`s. The length of the list must be {nparam}, boundary conditions must
    include initial conditions, `None` values will be replaced with some broad
    defaults. It is supported by MCMC only
ln_prior : str or list of ln_prior.LnPrior1D or None, optional
    Prior for MCMC, None means no prior. It is specified by a string literal
    or a list of {nparam} `ln_prior.LnPrior1D` objects, see `ln_prior`
    submodule for corresponding functions. Available string literals are:
    {ln_prior}
transform : bool or None, optional
    If `False` or `None` (default) output is not transformed. If `True` output
    is transformed as following:
     - Half-amplitude A is transformed as `zp - 2.5 lg(2*A)`, zp = 8.9,
       so that the amplitude is assumed to be the object peak flux in Jy.
     - baseline flux is normalised by A: baseline -> baseline / A
     - reference time is removed
     - goodness of fit is transformed as `ln(reduced chi^2 + 1)` to reduce
       its spread
     - other parameters are not transformed
    See `names` and `descriptions` attributes an object for the list and order
    of features.

{attr}
supported_algorithms : list of str
    Available argument values for the constructor

{methods}

{model}
Examples
--------
>>> import numpy as np
>>> from light_curve import {feature}
>>>
>>> fit = {feature}('mcmc')
>>> t = np.linspace(0, 10, 101)
>>> flux = 1 + (t - 3) ** 2
>>> fluxerr = np.sqrt(flux)
>>> result = fit(t, flux, fluxerr, sorted=True)
>>> # Result is built from a model parameters and reduced chi^2
>>> # So we can use as a `params` array of the static `.model()` method
>>> model = {feature}.model(t, result)
"#,
                    intro = <$eval>::doc().trim_start(),
                    names_descriptions = names_descriptions,
                    supported_algo = Self::supported_algorithms_str(),
                    mcmc_niter = lcf::McmcCurveFit::default_niterations(),
                    ceres_args = ceres_args,
                    lmsder_niter = lmsder_niter,
                    attr = ATTRIBUTES_DOC,
                    methods = METHODS_DOC,
                    model = FIT_METHOD_MODEL_DOC,
                    feature = stringify!($name),
                    nparam = $nparam,
                    ln_prior = $ln_prior_doc,
                )
            }
        }
    };
}

evaluator!(Amplitude, lcf::Amplitude, StockTransformer::Identity);

evaluator!(
    AndersonDarlingNormal,
    lcf::AndersonDarlingNormal,
    StockTransformer::Lg
);

#[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
pub struct BeyondNStd {}

impl_stock_transform!(BeyondNStd, StockTransformer::Identity);

#[pymethods]
impl BeyondNStd {
    #[new]
    #[pyo3(signature = (nstd=lcf::BeyondNStd::default_nstd(), *, transform=None))]
    fn __new__(nstd: f64, transform: Option<Bound<PyAny>>) -> Res<PyClassInitializer<Self>> {
        Ok(
            PyClassInitializer::from(PyFeatureEvaluator::with_py_transform(
                lcf::BeyondNStd::new(nstd as f32).into(),
                lcf::BeyondNStd::new(nstd).into(),
                transform,
                Self::DEFAULT_TRANSFORMER,
            )?)
            .add_subclass(Self {}),
        )
    }

    /// Required by pickle.load / pickle.loads
    #[staticmethod]
    fn __getnewargs__() -> (f64,) {
        (lcf::BeyondNStd::default_nstd(),)
    }

    #[classattr]
    fn __doc__() -> String {
        format!(
            r#"{header}

Parameters
----------
nstd : positive float
    N, default is {nstd_default:.1}
{transform}
{footer}"#,
            header = lcf::BeyondNStd::<f64>::doc().trim_start(),
            nstd_default = lcf::BeyondNStd::<f64>::default_nstd(),
            transform = transform_parameter_doc(Self::DEFAULT_TRANSFORMER),
            footer = COMMON_FEATURE_DOC,
        )
    }
}

fit_evaluator!(
    BazinFit,
    lcf::BazinFit,
    lcf::BazinInitsBounds,
    lcf::transformers::bazin_fit::BazinFitTransformer::default(),
    5,
    {
        "no" => lcf::BazinLnPrior::fixed(lcf::LnPrior::none()),
        s => return Err(Exception::ValueError(format!(
            "unsupported ln_prior name '{s}'"
        )).into()),
    },
    "'no': no prior",
);

#[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
pub struct Bins {}

#[pymethods]
impl Bins {
    #[new]
    #[pyo3(signature = (features, *, window, offset, transform = None))]
    fn __new__(
        features: Bound<PyAny>,
        window: f64,
        offset: f64,
        transform: Option<Bound<PyAny>>,
    ) -> PyResult<(Self, PyFeatureEvaluator)> {
        if transform.is_some() {
            return Err(Exception::NotImplementedError(
                "transform is not supported by Bins, apply transformations to individual features"
                    .to_string(),
            )
            .into());
        }
        let mut eval_f32 = lcf::Bins::default();
        let mut eval_f64 = lcf::Bins::default();
        for x in features.iter()? {
            let py_feature = x?.downcast::<PyFeatureEvaluator>()?.borrow();
            eval_f32.add_feature(py_feature.feature_evaluator_f32.clone());
            eval_f64.add_feature(py_feature.feature_evaluator_f64.clone());
        }

        eval_f32.set_window(window as f32);
        eval_f64.set_window(window);

        eval_f32.set_offset(offset as f32);
        eval_f64.set_offset(offset);

        Ok((
            Self {},
            PyFeatureEvaluator {
                feature_evaluator_f32: eval_f32.into(),
                feature_evaluator_f64: eval_f64.into(),
            },
        ))
    }

    /// Use __getnewargs_ex__ instead
    #[staticmethod]
    fn __getnewargs__() -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "use __getnewargs_ex__ instead",
        ))
    }

    /// Required by pickle.load / pickle.loads
    #[staticmethod]
    fn __getnewargs_ex__(py: Python) -> ((Bound<PyTuple>,), HashMap<&'static str, f64>) {
        (
            (PyTuple::empty_bound(py),),
            [
                ("window", lcf::Bins::<_, Feature<_>>::default_window()),
                ("offset", lcf::Bins::<_, Feature<_>>::default_offset()),
            ]
            .into(),
        )
    }

    #[classattr]
    fn __doc__() -> String {
        format!(
            r#"{header}

Parameters
----------
features : iterable
    Features to extract from binned time-series
window : positive float
    Width of binning interval in units of time
offset : float
    Zero time moment
transform : None
    Not supported, apply transformations to individual features
{footer}
"#,
            header = lcf::Bins::<f64, lcf::Feature<f64>>::doc().trim_start(),
            footer = COMMON_FEATURE_DOC,
        )
    }
}

evaluator!(Cusum, lcf::Cusum, StockTransformer::Identity);

evaluator!(Eta, lcf::Eta, StockTransformer::Identity);

evaluator!(EtaE, lcf::EtaE, StockTransformer::Lg);

evaluator!(
    ExcessVariance,
    lcf::ExcessVariance,
    StockTransformer::Identity
);

#[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
pub struct InterPercentileRange {}

impl_stock_transform!(InterPercentileRange, StockTransformer::Identity);

#[pymethods]
impl InterPercentileRange {
    #[new]
    #[pyo3(signature = (quantile=lcf::InterPercentileRange::default_quantile(), *, transform = None))]
    fn __new__(quantile: f32, transform: Option<Bound<PyAny>>) -> Res<PyClassInitializer<Self>> {
        Ok(
            PyClassInitializer::from(PyFeatureEvaluator::with_py_transform(
                lcf::InterPercentileRange::new(quantile).into(),
                lcf::InterPercentileRange::new(quantile).into(),
                transform,
                Self::DEFAULT_TRANSFORMER,
            )?)
            .add_subclass(Self {}),
        )
    }

    /// Required by pickle.load / pickle.loads
    #[staticmethod]
    fn __getnewargs__() -> (f32,) {
        (lcf::InterPercentileRange::default_quantile(),)
    }

    #[classattr]
    fn __doc__() -> String {
        format!(
            r#"{header}

Parameters
----------
quantile : positive float
    Range is (100% * quantile, 100% * (1 - quantile)). Default quantile is {quantile_default:.2}
{transform}
{footer}"#,
            header = lcf::InterPercentileRange::doc().trim_start(),
            quantile_default = lcf::InterPercentileRange::default_quantile(),
            transform = transform_parameter_doc(Self::DEFAULT_TRANSFORMER),
            footer = COMMON_FEATURE_DOC
        )
    }
}

evaluator!(Kurtosis, lcf::Kurtosis, StockTransformer::Arcsinh);

evaluator!(LinearFit, lcf::LinearFit, StockTransformer::Identity);

evaluator!(LinearTrend, lcf::LinearTrend, StockTransformer::Identity);

fit_evaluator!(
    LinexpFit,
    lcf::LinexpFit,
    lcf::LinexpInitsBounds,
    lcf::transformers::linexp_fit::LinexpFitTransformer::default(),
    4,
    {
        "no" => lcf::LinexpLnPrior::fixed(lcf::LnPrior::none()),
        s => return Err(Exception::ValueError(format!(
            "unsupported ln_prior name '{s}'"
        )).into()),
    },
    "'no': no prior",
);

#[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
pub struct MagnitudePercentageRatio {}

impl_stock_transform!(MagnitudePercentageRatio, StockTransformer::Identity);

#[pymethods]
impl MagnitudePercentageRatio {
    #[new]
    #[pyo3(signature = (
        quantile_numerator=lcf::MagnitudePercentageRatio::default_quantile_numerator(),
        quantile_denominator=lcf::MagnitudePercentageRatio::default_quantile_denominator(),
        *,
        transform=None,
    ))]
    fn __new__(
        quantile_numerator: f32,
        quantile_denominator: f32,
        transform: Option<Bound<PyAny>>,
    ) -> Res<PyClassInitializer<Self>> {
        if !(0.0..0.5).contains(&quantile_numerator) {
            return Err(Exception::ValueError(
                "quantile_numerator must be between 0.0 and 0.5".to_string(),
            ));
        }
        if !(0.0..0.5).contains(&quantile_denominator) {
            return Err(Exception::ValueError(
                "quantile_denumerator must be between 0.0 and 0.5".to_string(),
            ));
        }
        Ok(
            PyClassInitializer::from(PyFeatureEvaluator::with_py_transform(
                lcf::MagnitudePercentageRatio::new(quantile_numerator, quantile_denominator).into(),
                lcf::MagnitudePercentageRatio::new(quantile_numerator, quantile_denominator).into(),
                transform,
                Self::DEFAULT_TRANSFORMER,
            )?)
            .add_subclass(Self {}),
        )
    }

    /// Required by pickle.load / pickle.loads
    #[staticmethod]
    fn __getnewargs__() -> (f32, f32) {
        (
            lcf::MagnitudePercentageRatio::default_quantile_numerator(),
            lcf::MagnitudePercentageRatio::default_quantile_denominator(),
        )
    }

    #[classattr]
    fn __doc__() -> String {
        format!(
            r#"{header}

Parameters
----------
quantile_numerator: positive float
    Numerator is inter-percentile range (100% * q, 100% (1 - q)).
    Default value is {quantile_numerator_default:.2}
quantile_denominator: positive float
    Denominator is inter-percentile range (100% * q, 100% (1 - q)).
    Default value is {quantile_denominator_default:.2}
{transform}
{footer}"#,
            header = lcf::MagnitudePercentageRatio::doc().trim_start(),
            quantile_numerator_default =
                lcf::MagnitudePercentageRatio::default_quantile_numerator(),
            quantile_denominator_default =
                lcf::MagnitudePercentageRatio::default_quantile_denominator(),
            transform = transform_parameter_doc(Self::DEFAULT_TRANSFORMER),
            footer = COMMON_FEATURE_DOC
        )
    }
}

evaluator!(MaximumSlope, lcf::MaximumSlope, StockTransformer::ClippedLg);

evaluator!(Mean, lcf::Mean, StockTransformer::Identity);

evaluator!(MeanVariance, lcf::MeanVariance, StockTransformer::Identity);

evaluator!(Median, lcf::Median, StockTransformer::Identity);

evaluator!(
    MedianAbsoluteDeviation,
    lcf::MedianAbsoluteDeviation,
    StockTransformer::Identity
);

#[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
pub struct MedianBufferRangePercentage {}

impl_stock_transform!(MedianBufferRangePercentage, StockTransformer::Identity);

#[pymethods]
impl MedianBufferRangePercentage {
    #[new]
    #[pyo3(signature = (quantile=lcf::MedianBufferRangePercentage::<f64>::default_quantile(), *, transform = None))]
    fn __new__(quantile: f64, transform: Option<Bound<PyAny>>) -> Res<PyClassInitializer<Self>> {
        Ok(
            PyClassInitializer::from(PyFeatureEvaluator::with_py_transform(
                lcf::MedianBufferRangePercentage::new(quantile as f32).into(),
                lcf::MedianBufferRangePercentage::new(quantile).into(),
                transform,
                Self::DEFAULT_TRANSFORMER,
            )?)
            .add_subclass(Self {}),
        )
    }

    /// Required by pickle.load / pickle.loads
    #[staticmethod]
    fn __getnewargs__() -> (f64,) {
        (lcf::MedianBufferRangePercentage::default_quantile(),)
    }

    #[classattr]
    fn __doc__() -> String {
        format!(
            r#"{header}

Parameters
----------
quantile : positive float
    Relative range size, default is {quantile_default:.2}
{transform}
{footer}"#,
            header = lcf::MedianBufferRangePercentage::<f64>::doc(),
            quantile_default = lcf::MedianBufferRangePercentage::<f64>::default_quantile(),
            transform = transform_parameter_doc(Self::DEFAULT_TRANSFORMER),
            footer = COMMON_FEATURE_DOC
        )
    }
}

evaluator!(
    PercentAmplitude,
    lcf::PercentAmplitude,
    StockTransformer::Identity
);

#[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
pub struct PercentDifferenceMagnitudePercentile {}

impl_stock_transform!(
    PercentDifferenceMagnitudePercentile,
    StockTransformer::ClippedLg
);

#[pymethods]
impl PercentDifferenceMagnitudePercentile {
    #[new]
    #[pyo3(signature = (quantile=lcf::PercentDifferenceMagnitudePercentile::default_quantile(), *, transform = None))]
    fn __new__(quantile: f32, transform: Option<Bound<PyAny>>) -> Res<PyClassInitializer<Self>> {
        if !(0.0..0.5).contains(&quantile) {
            return Err(Exception::ValueError(
                "quantile must be between 0.0 and 0.5".to_string(),
            ));
        }
        Ok(
            PyClassInitializer::from(PyFeatureEvaluator::with_py_transform(
                lcf::PercentDifferenceMagnitudePercentile::new(quantile).into(),
                lcf::PercentDifferenceMagnitudePercentile::new(quantile).into(),
                transform,
                Self::DEFAULT_TRANSFORMER,
            )?)
            .add_subclass(Self {}),
        )
    }

    /// Required by pickle.load / pickle.loads
    #[staticmethod]
    fn __getnewargs__() -> (f32,) {
        (lcf::PercentDifferenceMagnitudePercentile::default_quantile(),)
    }

    #[classattr]
    fn __doc__() -> String {
        format!(
            r#"{header}

Parameters
----------
quantile : positive float
    Relative range size, default is {quantile_default:.2}
{transform}
{footer}"#,
            header = lcf::PercentDifferenceMagnitudePercentile::doc(),
            quantile_default = lcf::PercentDifferenceMagnitudePercentile::default_quantile(),
            transform = transform_parameter_doc(Self::DEFAULT_TRANSFORMER),
            footer = COMMON_FEATURE_DOC
        )
    }
}

type LcfPeriodogram<T> = lcf::Periodogram<T, lcf::Feature<T>>;

#[derive(FromPyObject)]
enum NyquistArgumentOfPeriodogram<'py> {
    String(&'py str),
    Float(f32),
}

#[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
pub struct Periodogram {
    eval_f32: LcfPeriodogram<f32>,
    eval_f64: LcfPeriodogram<f64>,
}

impl Periodogram {
    fn create_evals(
        peaks: Option<usize>,
        resolution: Option<f32>,
        max_freq_factor: Option<f32>,
        nyquist: Option<NyquistArgumentOfPeriodogram>,
        fast: Option<bool>,
        features: Option<Bound<PyAny>>,
    ) -> PyResult<(LcfPeriodogram<f32>, LcfPeriodogram<f64>)> {
        let mut eval_f32 = match peaks {
            Some(peaks) => lcf::Periodogram::new(peaks),
            None => lcf::Periodogram::default(),
        };
        let mut eval_f64 = match peaks {
            Some(peaks) => lcf::Periodogram::new(peaks),
            None => lcf::Periodogram::default(),
        };

        if let Some(resolution) = resolution {
            eval_f32.set_freq_resolution(resolution);
            eval_f64.set_freq_resolution(resolution);
        }
        if let Some(max_freq_factor) = max_freq_factor {
            eval_f32.set_max_freq_factor(max_freq_factor);
            eval_f64.set_max_freq_factor(max_freq_factor);
        }
        if let Some(nyquist) = nyquist {
            let nyquist_freq: lcf::NyquistFreq =
                match nyquist {
                    NyquistArgumentOfPeriodogram::String(nyquist_type) => match nyquist_type {
                        "average" => lcf::AverageNyquistFreq {}.into(),
                        "median" => lcf::MedianNyquistFreq {}.into(),
                        _ => return Err(PyValueError::new_err(
                            "nyquist must be one of: None, 'average', 'median' or quantile value",
                        )),
                    },
                    NyquistArgumentOfPeriodogram::Float(quantile) => {
                        lcf::QuantileNyquistFreq { quantile }.into()
                    }
                };
            eval_f32.set_nyquist(nyquist_freq.clone());
            eval_f64.set_nyquist(nyquist_freq);
        }
        if let Some(fast) = fast {
            if fast {
                eval_f32.set_periodogram_algorithm(lcf::PeriodogramPowerFft::new().into());
                eval_f64.set_periodogram_algorithm(lcf::PeriodogramPowerFft::new().into());
            } else {
                eval_f32.set_periodogram_algorithm(lcf::PeriodogramPowerDirect {}.into());
                eval_f64.set_periodogram_algorithm(lcf::PeriodogramPowerDirect {}.into());
            }
        }
        if let Some(features) = features {
            for x in features.iter()? {
                let py_feature = x?.downcast::<PyFeatureEvaluator>()?.borrow();
                eval_f32.add_feature(py_feature.feature_evaluator_f32.clone());
                eval_f64.add_feature(py_feature.feature_evaluator_f64.clone());
            }
        }
        Ok((eval_f32, eval_f64))
    }

    fn freq_power_impl<'py, T>(
        eval: &lcf::Periodogram<T, lcf::Feature<T>>,
        py: Python<'py>,
        t: Arr<T>,
        m: Arr<T>,
    ) -> (Bound<'py, PyUntypedArray>, Bound<'py, PyUntypedArray>)
    where
        T: lcf::Float + numpy::Element,
    {
        let t: DataSample<_> = t.as_array().into();
        let m: DataSample<_> = m.as_array().into();
        let mut ts = lcf::TimeSeries::new_without_weight(t, m);
        let (freq, power) = eval.freq_power(&mut ts);
        let freq = PyArray1::from_vec_bound(py, freq);
        let power = PyArray1::from_vec_bound(py, power);
        (freq.as_untyped().clone(), power.as_untyped().clone())
    }
}

#[pymethods]
impl Periodogram {
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (
        *,
        peaks = LcfPeriodogram::<f64>::default_peaks(),
        resolution = LcfPeriodogram::<f64>::default_resolution(),
        max_freq_factor = LcfPeriodogram::<f64>::default_max_freq_factor(),
        nyquist = NyquistArgumentOfPeriodogram::String("average"),
        fast = true,
        features = None,
        transform = None,
    ))]
    fn __new__(
        peaks: Option<usize>,
        resolution: Option<f32>,
        max_freq_factor: Option<f32>,
        nyquist: Option<NyquistArgumentOfPeriodogram>,
        fast: Option<bool>,
        features: Option<Bound<PyAny>>,
        transform: Option<Bound<PyAny>>,
    ) -> PyResult<(Self, PyFeatureEvaluator)> {
        if transform.is_some() {
            return Err(PyNotImplementedError::new_err(
                "transform is not supported by Periodogram, peak-related features are not transformed, but you still may apply transformation for the underlying features",
            ));
        }
        let (eval_f32, eval_f64) =
            Self::create_evals(peaks, resolution, max_freq_factor, nyquist, fast, features)?;
        Ok((
            Self {
                eval_f32: eval_f32.clone(),
                eval_f64: eval_f64.clone(),
            },
            PyFeatureEvaluator {
                feature_evaluator_f32: eval_f32.into(),
                feature_evaluator_f64: eval_f64.into(),
            },
        ))
    }

    /// Angular frequencies and periodogram values
    fn freq_power<'py>(
        &self,
        py: Python<'py>,
        t: Bound<PyAny>,
        m: Bound<PyAny>,
    ) -> Res<(Bound<'py, PyUntypedArray>, Bound<'py, PyUntypedArray>)> {
        dtype_dispatch!(
            |t, m| Ok(Self::freq_power_impl(&self.eval_f32, py, t, m)),
            |t, m| Ok(Self::freq_power_impl(&self.eval_f64, py, t, m)),
            t,
            =m
        )
    }

    #[classattr]
    fn __doc__() -> String {
        format!(
            r#"{intro}
Parameters
----------
peaks : int or None, optional
    Number of peaks to find, default is {default_peaks}
resolution : float or None, optional
    Resolution of frequency grid, default is {default_resolution}
max_freq_factor : float or None, optional
    Mulitplier for Nyquist frequency, default is {default_max_freq_factor}
nyquist : str or float or None, optional
    Type of Nyquist frequency. Could be one of:
     - 'average': "Average" Nyquist frequency
     - 'median': Nyquist frequency is defined by median time interval
        between observations
     - float: Nyquist frequency is defined by given quantile of time
        intervals between observations
    Default is '{default_nyquist}'
fast : bool or None, optional
    Use "Fast" (approximate and FFT-based) or direct periodogram algorithm,
    default is {default_fast}
features : iterable or None, optional
    Features to extract from periodogram considering it as a time-series,
    default is None which means no additional features
    Features to extract from periodogram considering it as a time-series
transform : None, optional
    Not supported for Periodogram, peaks are not transformed, but you still
    may apply transformation for the underlying features with thier
    constructors

{common}
freq_power(t, m)
    Get periodogram

    Parameters
    ----------
    t : np.ndarray of np.float32 or np.float64
        Time array
    m : np.ndarray of np.float32 or np.float64
        Magnitude (flux) array

    Returns
    -------
    freq : np.ndarray of np.float32 or np.float64
        Frequency grid
    power : np.ndarray of np.float32 or np.float64
        Periodogram power

Examples
--------
>>> import numpy as np
>>> from light_curve import Periodogram
>>> periodogram = Periodogram(peaks=2, resolution=20.0, max_freq_factor=2.0,
...                           nyquist='average', fast=True)
>>> t = np.linspace(0, 10, 101)
>>> m = np.sin(2*np.pi * t / 0.7) + 0.5 * np.cos(2*np.pi * t / 3.3)
>>> peaks = periodogram(t, m, sorted=True)[::2]
>>> frequency, power = periodogram.freq_power(t, m)
"#,
            intro = LcfPeriodogram::<f64>::doc(),
            default_peaks = LcfPeriodogram::<f64>::default_peaks(),
            default_resolution = LcfPeriodogram::<f64>::default_resolution(),
            default_max_freq_factor = LcfPeriodogram::<f64>::default_max_freq_factor(),
            default_nyquist = "average",
            default_fast = "True",
            common = ATTRIBUTES_DOC,
        )
    }
}

evaluator!(ReducedChi2, lcf::ReducedChi2, StockTransformer::Ln1p);

evaluator!(Roms, lcf::Roms, StockTransformer::Identity);

evaluator!(Skew, lcf::Skew, StockTransformer::Arcsinh);

evaluator!(
    StandardDeviation,
    lcf::StandardDeviation,
    StockTransformer::Identity
);

evaluator!(StetsonK, lcf::StetsonK, StockTransformer::Identity);

fit_evaluator!(
    VillarFit,
    lcf::VillarFit,
    lcf::VillarInitsBounds,
    lcf::transformers::villar_fit::VillarFitTransformer::default(),
    7,
    {
        "no" => lcf::VillarLnPrior::fixed(lcf::LnPrior::none()),
        "hosseinzadeh2020" => lcf::VillarLnPrior::hosseinzadeh2020(1.0, 0.0),
        s => return Err(Exception::ValueError(format!(
            "unsupported ln_prior name '{s}'"
        )).into()),
    },
    r"- 'no': no prior,\
    - 'hosseinzadeh2020': prior addopted from Hosseinzadeh et al. 2020, it
      assumes that `t` is in days",
);

evaluator!(WeightedMean, lcf::WeightedMean, StockTransformer::Identity);

evaluator!(Duration, lcf::Duration, StockTransformer::Identity);

evaluator!(
    MaximumTimeInterval,
    lcf::MaximumTimeInterval,
    StockTransformer::Identity
);

evaluator!(
    MinimumTimeInterval,
    lcf::MinimumTimeInterval,
    StockTransformer::Identity
);

evaluator!(
    ObservationCount,
    lcf::ObservationCount,
    StockTransformer::Identity
);

#[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
pub struct OtsuSplit {}

#[pymethods]
impl OtsuSplit {
    #[new]
    #[pyo3(signature = (*, transform=None))]
    fn __new__(transform: Option<Bound<PyAny>>) -> Res<(Self, PyFeatureEvaluator)> {
        if transform.is_some() {
            return Err(Exception::NotImplementedError(
                "OtsuSplit does not support transformations yet".to_string(),
            ));
        }
        Ok((
            Self {},
            PyFeatureEvaluator {
                feature_evaluator_f32: lcf::OtsuSplit::new().into(),
                feature_evaluator_f64: lcf::OtsuSplit::new().into(),
            },
        ))
    }

    #[staticmethod]
    fn threshold(m: Bound<PyAny>) -> Res<f64> {
        dtype_dispatch!({ Self::threshold_impl }(m))
    }

    #[classattr]
    fn __doc__() -> String {
        format!(
            "{}{}",
            lcf::OtsuSplit::doc().trim_start(),
            COMMON_FEATURE_DOC
        )
    }
}

impl OtsuSplit {
    fn threshold_impl<T>(m: Arr<T>) -> Res<f64>
    where
        T: lcf::Float + numpy::Element,
    {
        let mut ds = m.as_array().into();
        let (thr, _, _) = lcf::OtsuSplit::threshold(&mut ds).map_err(|_| {
            Exception::ValueError(
                "not enough points to find the threshold (minimum is 2)".to_string(),
            )
        })?;
        Ok(thr.value_as().unwrap())
    }
}

evaluator!(TimeMean, lcf::TimeMean, StockTransformer::Identity);

evaluator!(
    TimeStandardDeviation,
    lcf::TimeStandardDeviation,
    StockTransformer::Identity
);

/// Feature evaluator deserialized from JSON string
#[pyclass(name = "JSONDeserializedFeature", extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
pub struct JsonDeserializedFeature {}

#[pymethods]
impl JsonDeserializedFeature {
    #[new]
    #[pyo3(text_signature = "(json_string)")]
    fn __new__(s: String) -> Res<(Self, PyFeatureEvaluator)> {
        let feature_evaluator_f32: lcf::Feature<f32> = serde_json::from_str(&s).map_err(|err| {
            Exception::ValueError(format!("Cannot deserialize feature from JSON: {err}"))
        })?;
        let feature_evaluator_f64: lcf::Feature<f64> = serde_json::from_str(&s).map_err(|err| {
            Exception::ValueError(format!("Cannot deserialize feature from JSON: {err}"))
        })?;

        Ok((
            Self {},
            PyFeatureEvaluator {
                feature_evaluator_f32,
                feature_evaluator_f64,
            },
        ))
    }
}
