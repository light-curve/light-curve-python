use crate::check::{check_finite, check_no_nans, is_sorted};
use crate::cont_array::ContCowArray;
use crate::errors::{Exception, Res};
use crate::np_array::{Arr, GenericFloatArray1};

use const_format::formatcp;
use light_curve_feature::{self as lcf, prelude::*, DataSample};
use ndarray::IntoNdProducer;
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rayon::prelude::*;
use std::convert::{TryFrom, TryInto};

const ATTRIBUTES_DOC: &str = r#"Attributes
----------
names : list of str
    Feature names
descriptions : list of str
    Feature descriptions"#;

const METHOD_CALL_DOC: &str = r#"Methods
-------
__call__(t, m, sigma=None, sorted=None, check=True, fill_value=None)
    Extract features and return them as a numpy array

    Parameters
    ----------
    t : numpy.ndarray of np.float32 or np.float64 dtype
        Time moments
    m : numpy.ndarray of the same dtype as t
        Signal in magnitude or fluxes. Refer to the feature description to
        decide which would work better in your case
    sigma : numpy.ndarray of the same dtype as t, optional
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

const METHOD_MANY_DOC: &str = r#"
many(lcs, sorted=None, check=True, fill_value=None, n_jobs=-1)
    Parallel light curve feature extraction

    It is a parallel executed equivalent of
    >>> def many(lcs, sorted=None, fill_value=None):
    ...     return np.stack([feature(*lc, sorted=sorted, fill_value=fill_value)
    ...                      for lc in lcs])

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

const METHODS_DOC: &str = formatcp!(
    r#"Methods
-------
{}
{}"#,
    METHOD_CALL_DOC,
    METHOD_MANY_DOC,
);

const COMMON_FEATURE_DOC: &str = formatcp!("\n{}\n\n{}\n", ATTRIBUTES_DOC, METHODS_DOC);

type PyLightCurve<'a, T> = (Arr<'a, T>, Arr<'a, T>, Option<Arr<'a, T>>);

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
    fn call_impl<T>(
        feature_evaluator: &lcf::Feature<T>,
        t: Arr<T>,
        m: Arr<T>,
        sigma: Option<Arr<T>>,
        sorted: Option<bool>,
        check: bool,
        is_t_required: bool,
        fill_value: Option<T>,
    ) -> Res<ndarray::Array1<T>>
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
        Ok(result.into())
    }

    #[allow(clippy::too_many_arguments)]
    fn py_many<'a, T>(
        &self,
        feature_evaluator: &lcf::Feature<T>,
        py: Python,
        lcs: Vec<(
            GenericFloatArray1<'a>,
            GenericFloatArray1<'a>,
            Option<GenericFloatArray1<'a>>,
        )>,
        sorted: Option<bool>,
        check: bool,
        fill_value: Option<T>,
        n_jobs: i64,
    ) -> Res<PyObject>
    where
        T: lcf::Float + numpy::Element,
        Arr<'a, T>: TryFrom<GenericFloatArray1<'a>>,
    {
        let wrapped_lcs = lcs
            .into_iter()
            .enumerate()
            .map(|(i, (t, m, sigma))| {
                let t = TryInto::<Arr<_>>::try_into(t);
                let m = TryInto::<Arr<_>>::try_into(m);
                let sigma = sigma.map(TryInto::<Arr<_>>::try_into).transpose();

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
        .into_pyarray(py)
        .into_py(py))
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
    #[args(
        t,
        m,
        sigma = "None",
        sorted = "None",
        check = "true",
        fill_value = "None"
    )]
    fn __call__(
        &self,
        py: Python,
        t: GenericFloatArray1,
        m: GenericFloatArray1,
        sigma: Option<GenericFloatArray1>,
        sorted: Option<bool>,
        check: bool,
        fill_value: Option<f64>,
    ) -> Res<PyObject> {
        match (t, m) {
            (GenericFloatArray1::Float32(t), GenericFloatArray1::Float32(m)) => {
                let sigma = sigma
                    .map(|sigma| {
                        sigma.try_into().map_err(|_| {
                            Exception::ValueError(
                                "sigma is float64, but t & m are float32".to_string(),
                            )
                        })
                    })
                    .transpose()?;
                Ok(Self::call_impl(
                    &self.feature_evaluator_f32,
                    t,
                    m,
                    sigma,
                    sorted,
                    check,
                    self.is_t_required(sorted),
                    fill_value.map(|v| v as f32),
                )?
                .into_pyarray(py)
                .into_py(py))
            }
            (GenericFloatArray1::Float64(t), GenericFloatArray1::Float64(m)) => {
                let sigma = sigma
                    .map(|sigma| {
                        sigma.try_into().map_err(|_| {
                            Exception::ValueError(
                                "sigma is float32, but t & m are float64".to_string(),
                            )
                        })
                    })
                    .map_or(Ok(None), |result| result.map(Some))?;
                Ok(Self::call_impl(
                    &self.feature_evaluator_f64,
                    t,
                    m,
                    sigma,
                    sorted,
                    check,
                    self.is_t_required(sorted),
                    fill_value,
                )?
                .into_pyarray(py)
                .into_py(py))
            }
            _ => Err(Exception::ValueError("t and m have different dtype".into())),
        }
    }

    #[args(lcs, sorted = "None", check = "true", fill_value = "None", n_jobs = -1)]
    fn many(
        &self,
        py: Python,
        lcs: Vec<(
            GenericFloatArray1,
            GenericFloatArray1,
            Option<GenericFloatArray1>,
        )>,
        sorted: Option<bool>,
        check: bool,
        fill_value: Option<f64>,
        n_jobs: i64,
    ) -> Res<PyObject> {
        if lcs.is_empty() {
            Err(Exception::ValueError("lcs is empty".to_string()))
        } else {
            match lcs[0].0 {
                GenericFloatArray1::Float32(_) => self.py_many(
                    &self.feature_evaluator_f32,
                    py,
                    lcs,
                    sorted,
                    check,
                    fill_value.map(|v| v as f32),
                    n_jobs,
                ),
                GenericFloatArray1::Float64(_) => self.py_many(
                    &self.feature_evaluator_f64,
                    py,
                    lcs,
                    sorted,
                    check,
                    fill_value,
                    n_jobs,
                ),
            }
        }
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
}

#[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
#[pyo3(text_signature = "(*features)")]
pub struct Extractor {}

#[pymethods]
impl Extractor {
    #[new]
    #[args(args = "*")]
    fn __new__(args: &PyTuple) -> PyResult<(Self, PyFeatureEvaluator)> {
        let evals_iter = args.iter().map(|arg| {
            arg.downcast::<PyCell<PyFeatureEvaluator>>().map(|fe| {
                let fe = fe.borrow();
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
{}
"#,
            lcf::FeatureExtractor::<f64, lcf::Feature<f64>>::doc().trim_start(),
            COMMON_FEATURE_DOC,
        )
    }
}

macro_rules! evaluator {
    ($name: ident, $eval: ty $(,)?) => {
        #[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
        #[pyo3(text_signature = "()")]
        pub struct $name {}

        #[pymethods]
        impl $name {
            #[new]
            fn __new__() -> (Self, PyFeatureEvaluator) {
                (
                    Self {},
                    PyFeatureEvaluator {
                        feature_evaluator_f32: <$eval>::new().into(),
                        feature_evaluator_f64: <$eval>::new().into(),
                    },
                )
            }

            #[classattr]
            fn __doc__() -> String {
                format!("{}{}", <$eval>::doc().trim_start(), COMMON_FEATURE_DOC)
            }
        }
    };
}

const N_ALGO_CURVE_FIT: usize = {
    #[cfg(feature = "gsl")]
    {
        3
    }
    #[cfg(not(feature = "gsl"))]
    {
        1
    }
};

const SUPPORTED_ALGORITHMS_CURVE_FIT: [&str; N_ALGO_CURVE_FIT] = [
    "mcmc",
    #[cfg(feature = "gsl")]
    "lmsder",
    #[cfg(feature = "gsl")]
    "mcmc-lmsder",
];

macro_rules! fit_evaluator {
    ($name: ident, $eval: ty, $ib: ty, $nparam: literal $(,)?) => {
        #[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
        #[pyo3(text_signature = "(algorithm, mcmc_niter=None, lmsder_niter=None, init=None, bounds=None)")]
        pub struct $name {}

        impl $name {
            fn supported_algorithms_str() -> String {
                return SUPPORTED_ALGORITHMS_CURVE_FIT.join(", ");
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
        }

        #[pymethods]
        impl $name {
            #[new]
            #[args(
                algorithm,
                mcmc_niter = "None",
                lmsder_niter = "None",
                init = "None",
                bounds = "None"
            )]
            fn __new__(
                algorithm: &str,
                mcmc_niter: Option<u32>,
                lmsder_niter: Option<u16>,
                init: Option<Vec<Option<f64>>>,
                bounds: Option<Vec<(Option<f64>, Option<f64>)>>,
            ) -> PyResult<(Self, PyFeatureEvaluator)> {
                let mcmc_niter = mcmc_niter.unwrap_or_else(lcf::McmcCurveFit::default_niterations);

                #[cfg(feature = "gsl")]
                let lmsder_fit: lcf::CurveFitAlgorithm = lcf::LmsderCurveFit::new(
                    lmsder_niter.unwrap_or_else(lcf::LmsderCurveFit::default_niterations),
                )
                .into();
                #[cfg(not(feature = "gsl"))]
                if lmsder_niter.is_some() {
                    return Err(PyValueError::new_err(
                        "Compiled without GSL support, lmsder_niter is not supported",
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

                let curve_fit_algorithm: lcf::CurveFitAlgorithm = match algorithm {
                    "mcmc" => lcf::McmcCurveFit::new(mcmc_niter, None).into(),
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

                Ok((
                    Self {},
                    PyFeatureEvaluator {
                        feature_evaluator_f32: <$eval>::new(
                            curve_fit_algorithm.clone(),
                            <$eval>::default_ln_prior(),
                            init_bounds.clone(),
                        )
                        .into(),
                        feature_evaluator_f64: <$eval>::new(
                            curve_fit_algorithm,
                            <$eval>::default_ln_prior(),
                            init_bounds,
                        )
                        .into(),
                    },
                ))
            }

            #[staticmethod]
            #[args(t, params)]
            fn model(
                py: Python,
                t: GenericFloatArray1,
                params: GenericFloatArray1,
            ) -> Res<PyObject> {
                match (t, params) {
                    (GenericFloatArray1::Float32(t), GenericFloatArray1::Float32(params)) => {
                        Ok(Self::model_impl(t, params).into_pyarray(py).into_py(py))
                    }
                    (GenericFloatArray1::Float64(t), GenericFloatArray1::Float64(params)) => {
                        Ok(Self::model_impl(t, params).into_pyarray(py).into_py(py))
                    }
                    _ => Err(Exception::ValueError(
                        "t and params must have the same dtype".to_string(),
                    )),
                }
            }

            #[classattr]
            fn supported_algorithms() -> [&'static str; N_ALGO_CURVE_FIT] {
                return SUPPORTED_ALGORITHMS_CURVE_FIT;
            }

            #[classattr]
            fn __doc__() -> String {
                #[cfg(feature = "gsl")]
                let lmsder_niter = format!(
                    r#"lmsder_niter : int, optional
    Number of LMSDER iterations, default is {}
"#,
                    lcf::LmsderCurveFit::default_niterations()
                );
                #[cfg(not(feature = "gsl"))]
                let lmsder_niter = "";

                format!(
                    r#"{intro}
Parameters
----------
algorithm : str
    Non-linear least-square algorithm, supported values are:
    {supported_algo}.
mcmc_niter : int, optional
    Number of MCMC iterations, default is {mcmc_niter}
{lmsder_niter}init : list or None, optional
    Initial conditions, must be `None` or a `list` of `float`s or `None`s.
    The length of the list must be {nparam}, `None` values will be replaced
    with some defauls values. It is supported by MCMC only
bounds : list of tuples or None, optional
    Boundary conditions, must be `None` or a `list` of `tuple`s of `float`s or
    `None`s. The length of the list must be {nparam}, boundary conditions must
    include initial conditions, `None` values will be replaced with some broad
    defaults. It is supported by MCMC only

{attr}
supported_algorithms : list of str
    Available argument values for the constructor

{methods}

model(t, params)
    Underlying parametric model function

    Parameters
    ----------
    t : np.ndarray of np.float32 or np.float64
        Time moments, can be unsorted
    params : np.ndaarray of np.float32 or np.float64
        Parameters of the model, this array can be longer than actual parameter
        list, the beginning part of the array will be used in this case

    Returns
    -------
    np.ndarray of np.float32 or np.float64
        Array of model values corresponded to the given time moments

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
>>> # So we can use as a `params` array
>>> model = {feature}.model(t, result)
"#,
                    intro = <$eval>::doc().trim_start(),
                    supported_algo = Self::supported_algorithms_str(),
                    mcmc_niter = lcf::McmcCurveFit::default_niterations(),
                    lmsder_niter = lmsder_niter,
                    attr = ATTRIBUTES_DOC,
                    methods = METHODS_DOC,
                    feature = stringify!($name),
                    nparam = $nparam,
                )
            }
        }
    };
}

evaluator!(Amplitude, lcf::Amplitude);

evaluator!(AndersonDarlingNormal, lcf::AndersonDarlingNormal);

#[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
#[pyo3(text_signature = "(nstd, /)")]
pub struct BeyondNStd {}

#[pymethods]
impl BeyondNStd {
    #[new]
    #[args(nstd)]
    fn __new__(nstd: f64) -> (Self, PyFeatureEvaluator) {
        (
            Self {},
            PyFeatureEvaluator {
                feature_evaluator_f32: lcf::BeyondNStd::new(nstd as f32).into(),
                feature_evaluator_f64: lcf::BeyondNStd::new(nstd).into(),
            },
        )
    }

    #[classattr]
    fn __doc__() -> String {
        format!(
            r#"{}

Parameters
----------
nstd : positive float
    N
{}"#,
            lcf::BeyondNStd::<f64>::doc().trim_start(),
            COMMON_FEATURE_DOC,
        )
    }
}

fit_evaluator!(BazinFit, lcf::BazinFit, lcf::BazinInitsBounds, 5);

#[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
#[pyo3(text_signature = "(features, window, offset)")]
pub struct Bins {}

#[pymethods]
impl Bins {
    #[new]
    #[args(features, window, offset)]
    fn __new__(
        py: Python,
        features: PyObject,
        window: f64,
        offset: f64,
    ) -> PyResult<(Self, PyFeatureEvaluator)> {
        let mut eval_f32 = lcf::Bins::default();
        let mut eval_f64 = lcf::Bins::default();
        for x in features.extract::<&PyAny>(py)?.iter()? {
            let py_feature = x?.downcast::<PyCell<PyFeatureEvaluator>>()?.borrow();
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

    #[classattr]
    fn __doc__() -> String {
        format!(
            r#"{}

Parameters
----------
features : iterable
    Features to extract from binned time-series
window : positive float
    Width of binning interval in units of time
offset : float
    Zero time moment
"#,
            lcf::Bins::<f64, lcf::Feature<f64>>::doc().trim_start()
        )
    }
}

evaluator!(Cusum, lcf::Cusum);

evaluator!(Eta, lcf::Eta);

evaluator!(EtaE, lcf::EtaE);

evaluator!(ExcessVariance, lcf::ExcessVariance);

#[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
#[pyo3(text_signature = "(quantile)")]
pub struct InterPercentileRange {}

#[pymethods]
impl InterPercentileRange {
    #[new]
    #[args(quantile)]
    fn __new__(quantile: f32) -> (Self, PyFeatureEvaluator) {
        (
            Self {},
            PyFeatureEvaluator {
                feature_evaluator_f32: lcf::InterPercentileRange::new(quantile).into(),
                feature_evaluator_f64: lcf::InterPercentileRange::new(quantile).into(),
            },
        )
    }

    #[classattr]
    fn __doc__() -> String {
        format!(
            r#"{}

Parameters
----------
quantile : positive float
    Range is (100% * quantile, 100% * (1 - quantile))
{}"#,
            lcf::InterPercentileRange::doc().trim_start(),
            COMMON_FEATURE_DOC
        )
    }
}

evaluator!(Kurtosis, lcf::Kurtosis);

evaluator!(LinearFit, lcf::LinearFit);

evaluator!(LinearTrend, lcf::LinearTrend);

#[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
#[pyo3(text_signature = "(quantile_numerator, quantile_denominator)")]
pub struct MagnitudePercentageRatio {}

#[pymethods]
impl MagnitudePercentageRatio {
    #[new]
    #[args(quantile_numerator, quantile_denominator)]
    fn __new__(
        quantile_numerator: f32,
        quantile_denominator: f32,
    ) -> PyResult<(Self, PyFeatureEvaluator)> {
        if !(0.0..0.5).contains(&quantile_numerator) {
            return Err(PyValueError::new_err(
                "quantile_numerator must be between 0.0 and 0.5",
            ));
        }
        if !(0.0..0.5).contains(&quantile_denominator) {
            return Err(PyValueError::new_err(
                "quantile_denumerator must be between 0.0 and 0.5",
            ));
        }
        Ok((
            Self {},
            PyFeatureEvaluator {
                feature_evaluator_f32: lcf::MagnitudePercentageRatio::new(
                    quantile_numerator,
                    quantile_denominator,
                )
                .into(),
                feature_evaluator_f64: lcf::MagnitudePercentageRatio::new(
                    quantile_numerator,
                    quantile_denominator,
                )
                .into(),
            },
        ))
    }

    #[classattr]
    fn __doc__() -> String {
        format!(
            r#"{}

Parameters
----------
quantile_numerator: positive float
    Numerator is inter-percentile range (100% * q, 100% (1 - q))
quantile_denominator: positive float
    Denominator is inter-percentile range (100% * q, 100% (1 - q))
{}"#,
            lcf::MagnitudePercentageRatio::doc().trim_start(),
            COMMON_FEATURE_DOC
        )
    }
}

evaluator!(MaximumSlope, lcf::MaximumSlope);

evaluator!(Mean, lcf::Mean);

evaluator!(MeanVariance, lcf::MeanVariance);

evaluator!(Median, lcf::Median);

evaluator!(MedianAbsoluteDeviation, lcf::MedianAbsoluteDeviation,);

#[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
#[pyo3(text_signature = "(quantile)")]
pub struct MedianBufferRangePercentage {}

#[pymethods]
impl MedianBufferRangePercentage {
    #[new]
    #[args(quantile)]
    fn __new__(quantile: f64) -> (Self, PyFeatureEvaluator) {
        (
            Self {},
            PyFeatureEvaluator {
                feature_evaluator_f32: lcf::MedianBufferRangePercentage::new(quantile as f32)
                    .into(),
                feature_evaluator_f64: lcf::MedianBufferRangePercentage::new(quantile).into(),
            },
        )
    }

    #[classattr]
    fn __doc__() -> String {
        format!(
            r#"{}

Parameters
----------
quantile : positive float
    Relative range size
{}"#,
            lcf::MedianBufferRangePercentage::<f64>::doc(),
            COMMON_FEATURE_DOC
        )
    }
}

evaluator!(PercentAmplitude, lcf::PercentAmplitude);

#[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
#[pyo3(text_signature = "(quantile)")]
pub struct PercentDifferenceMagnitudePercentile {}

#[pymethods]
impl PercentDifferenceMagnitudePercentile {
    #[new]
    #[args(quantile)]
    fn __new__(quantile: f32) -> (Self, PyFeatureEvaluator) {
        (
            Self {},
            PyFeatureEvaluator {
                feature_evaluator_f32: lcf::PercentDifferenceMagnitudePercentile::new(quantile)
                    .into(),
                feature_evaluator_f64: lcf::PercentDifferenceMagnitudePercentile::new(quantile)
                    .into(),
            },
        )
    }

    #[classattr]
    fn __doc__() -> String {
        format!(
            r#"{}

Parameters
----------
quantile : positive float
    Relative range size
{}"#,
            lcf::PercentDifferenceMagnitudePercentile::doc(),
            COMMON_FEATURE_DOC
        )
    }
}

type LcfPeriodogram<T> = lcf::Periodogram<T, lcf::Feature<T>>;

#[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
#[pyo3(
    text_signature = "(peaks=None, resolution=None, max_freq_factor=None, nyquist=None, fast=None, features=None)"
)]
pub struct Periodogram {
    eval_f32: LcfPeriodogram<f32>,
    eval_f64: LcfPeriodogram<f64>,
}

impl Periodogram {
    fn create_evals(
        py: Python,
        peaks: Option<usize>,
        resolution: Option<f32>,
        max_freq_factor: Option<f32>,
        nyquist: Option<PyObject>,
        fast: Option<bool>,
        features: Option<PyObject>,
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
                if let Ok(s) = nyquist.extract::<&str>(py) {
                    match s {
                        "average" => lcf::AverageNyquistFreq {}.into(),
                        "median" => lcf::MedianNyquistFreq {}.into(),
                        _ => return Err(PyValueError::new_err(
                            "nyquist must be one of: None, 'average', 'median' or quantile value",
                        )),
                    }
                } else if let Ok(quantile) = nyquist.extract::<f32>(py) {
                    lcf::QuantileNyquistFreq { quantile }.into()
                } else {
                    return Err(PyValueError::new_err(
                        "nyquist must be one of: None, 'average', 'median' or quantile value",
                    ));
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
            for x in features.extract::<&PyAny>(py)?.iter()? {
                let py_feature = x?.downcast::<PyCell<PyFeatureEvaluator>>()?.borrow();
                eval_f32.add_feature(py_feature.feature_evaluator_f32.clone());
                eval_f64.add_feature(py_feature.feature_evaluator_f64.clone());
            }
        }
        Ok((eval_f32, eval_f64))
    }

    fn freq_power_impl<T>(
        eval: &lcf::Periodogram<T, lcf::Feature<T>>,
        t: Arr<T>,
        m: Arr<T>,
    ) -> (ndarray::Array1<T>, ndarray::Array1<T>)
    where
        T: lcf::Float + numpy::Element,
    {
        let t: DataSample<_> = t.as_array().into();
        let m: DataSample<_> = m.as_array().into();
        let mut ts = lcf::TimeSeries::new_without_weight(t, m);
        let (freq, power) = eval.freq_power(&mut ts);
        (freq.into(), power.into())
    }
}

#[pymethods]
impl Periodogram {
    #[new]
    #[args(
        peaks = "None",
        resolution = "None",
        max_freq_factor = "None",
        nyquist = "None",
        fast = "None",
        features = "None"
    )]
    fn __new__(
        py: Python,
        peaks: Option<usize>,
        resolution: Option<f32>,
        max_freq_factor: Option<f32>,
        nyquist: Option<PyObject>,
        fast: Option<bool>,
        features: Option<PyObject>,
    ) -> PyResult<(Self, PyFeatureEvaluator)> {
        let (eval_f32, eval_f64) = Self::create_evals(
            py,
            peaks,
            resolution,
            max_freq_factor,
            nyquist,
            fast,
            features,
        )?;
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
    #[pyo3(text_signature = "(t, m)")]
    fn freq_power(
        &self,
        py: Python,
        t: GenericFloatArray1,
        m: GenericFloatArray1,
    ) -> Res<(PyObject, PyObject)> {
        match (t, m) {
            (GenericFloatArray1::Float32(t), GenericFloatArray1::Float32(m)) => {
                let (freq, power) = Self::freq_power_impl(&self.eval_f32, t, m);
                Ok((
                    freq.into_pyarray(py).into_py(py),
                    power.into_pyarray(py).into_py(py),
                ))
            }
            (GenericFloatArray1::Float64(t), GenericFloatArray1::Float64(m)) => {
                let (freq, power) = Self::freq_power_impl(&self.eval_f64, t, m);
                Ok((
                    freq.into_pyarray(py).into_py(py),
                    power.into_pyarray(py).into_py(py),
                ))
            }
            _ => Err(Exception::ValueError(
                "t and m must have the same dtype".to_string(),
            )),
        }
    }

    #[classattr]
    fn __doc__() -> String {
        format!(
            r#"{intro}
Parameters
----------
peaks : int or None, optional
    Number of peaks to find

resolution : float or None, optional
    Resolution of frequency grid

max_freq_factor : float or None, optional
    Mulitplier for Nyquist frequency

nyquist : str or float or None, optional
    Type of Nyquist frequency. Could be one of:
     - 'average': "Average" Nyquist frequency
     - 'median': Nyquist frequency is defined by median time interval
        between observations
     - float: Nyquist frequency is defined by given quantile of time
        intervals between observations

fast : bool or None, optional
    Use "Fast" (approximate and FFT-based) or direct periodogram algorithm

features : iterable or None, optional
    Features to extract from periodogram considering it as a time-series

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
            intro = lcf::Periodogram::<f64, lcf::Feature<f64>>::doc(),
            common = ATTRIBUTES_DOC,
        )
    }
}

evaluator!(ReducedChi2, lcf::ReducedChi2);

evaluator!(Skew, lcf::Skew);

evaluator!(StandardDeviation, lcf::StandardDeviation);

evaluator!(StetsonK, lcf::StetsonK);

fit_evaluator!(VillarFit, lcf::VillarFit, lcf::VillarInitsBounds, 7);

evaluator!(WeightedMean, lcf::WeightedMean);

evaluator!(Duration, lcf::Duration);

evaluator!(MaximumTimeInterval, lcf::MaximumTimeInterval);

evaluator!(MinimumTimeInterval, lcf::MinimumTimeInterval);

evaluator!(ObservationCount, lcf::ObservationCount);

evaluator!(TimeMean, lcf::TimeMean);

evaluator!(TimeStandardDeviation, lcf::TimeStandardDeviation);
