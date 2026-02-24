use crate::arrow_input::{
    ArrowDtype, ArrowFloat, ArrowLcsSchema, ArrowListType, validate_arrow_lcs,
};
use crate::check::{check_finite, check_no_nans, is_sorted};
use crate::cont_array::ContCowArray;
use crate::errors::{Exception, Res};
use crate::ln_prior::LnPrior1D;
use crate::np_array::Arr;
use crate::transform::{StockTransformer, parse_transform};

use arrow_array::Array;
use arrow_array::cast::AsArray;
use const_format::formatcp;
use conv::ConvUtil;
use itertools::Itertools;
use light_curve_feature::{
    self as lcf, DataSample,
    periodogram::{FreqGrid, PeriodogramNormalization},
    prelude::*,
};
use macro_const::macro_const;
use ndarray::IntoNdProducer;
use num_traits::Zero;
use numpy::prelude::*;
use numpy::{AllowTypeChange, PyArray1, PyArrayLike1, PyUntypedArray};
use once_cell::sync::OnceCell;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyTuple};
use pyo3_arrow::PyChunkedArray;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::TryInto;
use std::ops::Deref;
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

const METHOD_CALL_DOC: &str = r#"__call__(self, t, m, sigma=None, *, fill_value=None, sorted=None, check=True, cast=False)
    Extract features and return them as a numpy array

    Parameters
    ----------
    t : numpy.ndarray of np.float32 or np.float64 dtype
        Time moments
    m : numpy.ndarray
        Signal in magnitude or fluxes. Refer to the feature description to
        decide which would work better in your case
    sigma : numpy.ndarray, optional
        Observation error, if None it is assumed to be unity
    fill_value : float or None, optional
        Value to fill invalid feature values, for example if count of
        observations is not enough to find a proper value.
        None causes exception for invalid features
    sorted : bool or None, optional
        Specifies if input array are sorted by time moments.
        True is for certainly sorted, False is for unsorted.
        If None is specified than sorting is checked and an exception is
        raised for unsorted `t`
    check : bool, optional
        Check all input arrays for NaNs, `t` and `m` for infinite values
    cast : bool, optional
        Allows non-numpy input and casting of arrays to a common dtype.
        If `False`, inputs must be `np.ndarray` instances with matched dtypes.
        Casting provides more flexibility with input types at the cost of
        performance.
    Returns
    -------
    ndarray of np.float32 or np.float64
        Extracted feature array"#;

macro_const! {
    const METHOD_MANY_DOC: &str = r#"
many(self, lcs, *, fill_value=None, sorted=None, check=True, cast=False, n_jobs=-1)
    Parallel light curve feature extraction

    It is a parallel executed equivalent of
    >>> def many(self, lcs, *, fill_value=None, sorted=None, check=True):
    ...     return np.stack(
    ...         [
    ...             self(
    ...                 *lc,
    ...                 fill_value=fill_value,
    ...                 sorted=sorted,
    ...                 check=check,
    ...                 cast=False,
    ...             )
    ...             for lc in lcs
    ...         ]
    ...     )

    Parameters
    ----------
    lcs : list of (t, m, sigma) or Arrow array
        Either a list of light curves packed into three-tuples (all numpy.ndarray
        of the same dtype), or an Arrow array/chunked array of type
        List<Struct<t, m[, sigma]>> where all fields share the same float dtype
        (float32 or float64). Arrow input is auto-detected via the
        __arrow_c_array__ / __arrow_c_stream__ protocol and enables zero-copy
        data access from pyarrow, polars, and other Arrow-compatible libraries.
        When using Arrow input, 2 struct fields means (t, m) without sigma,
        3 fields means (t, m, sigma). Field names are ignored
    fill_value : float or None, optional
        Fill invalid values by this or raise an exception if None
    sorted : bool or None, optional
        Specifies if input array are sorted by time moments, see __call__
        documentation for details
    check : bool, optional
        Check all input arrays for NaNs, `t` and `m` for infinite values
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
    )
}

type PyLightCurve<'a, T> = (Arr<'a, T>, Arr<'a, T>, Option<Arr<'a, T>>);

#[derive(Serialize, Deserialize, Clone)]
#[pyclass(
    subclass,
    name = "_FeatureEvaluator",
    module = "light_curve.light_curve_ext",
    from_py_object
)]
pub struct PyFeatureEvaluator {
    feature_evaluator_f32: Feature<f32>,
    feature_evaluator_f64: Feature<f64>,
}

impl PyFeatureEvaluator {
    fn with_transform(
        (fe_f32, fe_f64): (Feature<f32>, Feature<f64>),
        (tr_f32, tr_f64): (lcf::Transformer<f32>, lcf::Transformer<f64>),
    ) -> Res<Self> {
        Ok(Self {
            feature_evaluator_f32: lcf::Transformed::new(fe_f32, tr_f32)
                .map_err(|err| {
                    Exception::ValueError(format!(
                        "feature and transformation are incompatible: {err:?}"
                    ))
                })?
                .into(),
            feature_evaluator_f64: lcf::Transformed::new(fe_f64, tr_f64)
                .map_err(|err| {
                    Exception::ValueError(format!(
                        "feature and transformation are incompatible: {err:?}"
                    ))
                })?
                .into(),
        })
    }

    fn with_py_transform(
        fe_f32: Feature<f32>,
        fe_f64: Feature<f64>,
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

    /// Core TimeSeries construction from array views.
    /// Used by both the numpy and arrow input paths.
    fn ts_from_views<'a, T>(
        feature_evaluator: &Feature<T>,
        t: ndarray::ArrayView1<'a, T>,
        m: ndarray::ArrayView1<'a, T>,
        sigma: Option<ndarray::ArrayView1<'a, T>>,
        sorted: Option<bool>,
        check: bool,
        is_t_required: bool,
    ) -> Res<lcf::TimeSeries<'a, T>>
    where
        T: Float,
    {
        // Check finite on the views as passed by the caller. For numpy,
        // ts_from_numpy passes a unity broadcast when !required && !contiguous,
        // so this is a harmless no-op in that case.
        if check {
            check_finite(t)?;
            check_finite(m)?;
        }

        let mut t_ds: lcf::DataSample<_> = if is_t_required {
            t.into()
        } else {
            T::array0_unity().broadcast(t.len()).unwrap().into()
        };
        match sorted {
            Some(true) => {}
            Some(false) => {
                return Err(Exception::NotImplementedError(
                    "sorting is not implemented, please provide time-sorted arrays".to_string(),
                ));
            }
            None => {
                if feature_evaluator.is_sorting_required() && !is_sorted(t_ds.as_slice()) {
                    return Err(Exception::ValueError(
                        "t must be in ascending order".to_string(),
                    ));
                }
            }
        }

        let m_ds: lcf::DataSample<_> = if feature_evaluator.is_m_required() {
            m.into()
        } else {
            T::array0_unity().broadcast(m.len()).unwrap().into()
        };

        let w = match sigma {
            Some(sigma_view) if feature_evaluator.is_w_required() => {
                if check {
                    check_no_nans(sigma_view)?;
                }
                let mut a = sigma_view.to_owned();
                a.mapv_inplace(|x| x.powi(-2));
                Some(a)
            }
            _ => None,
        };

        let ts = match w {
            Some(w) => lcf::TimeSeries::new(t_ds, m_ds, w),
            None => lcf::TimeSeries::new_without_weight(t_ds, m_ds),
        };

        Ok(ts)
    }

    fn ts_from_numpy<'a, T>(
        feature_evaluator: &Feature<T>,
        t: &'a Arr<'a, T>,
        m: &'a Arr<'a, T>,
        sigma: &'a Option<Arr<'a, T>>,
        sorted: Option<bool>,
        check: bool,
        is_t_required: bool,
    ) -> Res<lcf::TimeSeries<'a, T>>
    where
        T: Float + numpy::Element,
    {
        if t.len() != m.len() {
            return Err(Exception::ValueError(
                "t and m must have the same size".to_string(),
            ));
        }
        if let Some(sigma) = sigma {
            if t.len() != sigma.len() {
                return Err(Exception::ValueError(
                    "t and sigma must have the same size".to_string(),
                ));
            }
        }

        // For non-contiguous numpy arrays that aren't needed, use the actual
        // array view anyway — ts_from_views will substitute unity when the
        // feature doesn't require the data. For contiguous arrays or when
        // required, extract the real view. We still pass is_t_required through
        // so ts_from_views can decide whether to check/use the data.
        let t_view = if is_t_required || t.is_contiguous() {
            t.as_array()
        } else {
            T::array0_unity().broadcast(t.len()).unwrap()
        };
        let m_view = if feature_evaluator.is_m_required() || m.is_contiguous() {
            m.as_array()
        } else {
            T::array0_unity().broadcast(m.len()).unwrap()
        };
        let sigma_view = sigma.as_ref().map(|s| s.as_array());

        Self::ts_from_views(
            feature_evaluator,
            t_view,
            m_view,
            sigma_view,
            sorted,
            check,
            is_t_required,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn call_impl<'py, T>(
        feature_evaluator: &Feature<T>,
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
        T: Float + numpy::Element,
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
        let array = PyArray1::from_vec(py, result);
        Ok(array.as_untyped().clone())
    }

    #[allow(clippy::too_many_arguments)]
    fn py_many<'py, T>(
        &self,
        feature_evaluator: &Feature<T>,
        py: Python<'py>,
        lcs: PyLcs<'py>,
        sorted: Option<bool>,
        check: bool,
        fill_value: Option<T>,
        n_jobs: i64,
    ) -> Res<Bound<'py, PyUntypedArray>>
    where
        T: Float + numpy::Element,
    {
        let wrapped_lcs = lcs
            .into_iter()
            .enumerate()
            .map(|(i, (t, m, sigma))| {
                let t = t.cast::<PyArray1<T>>().map(|a| a.readonly());
                let m = m.cast::<PyArray1<T>>().map(|a| a.readonly());
                let sigma = match &sigma {
                    Some(sigma) => sigma.cast::<PyArray1<T>>().map(|a| Some(a.readonly())),
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
        .into_pyarray(py)
        .as_untyped()
        .clone())
    }

    /// Parallel feature evaluation over a pre-built vector of TimeSeries.
    fn eval_many_parallel<T: Float>(
        feature_evaluator: &Feature<T>,
        mut tss: Vec<lcf::TimeSeries<'_, T>>,
        fill_value: Option<T>,
        n_jobs: i64,
    ) -> Res<ndarray::Array2<T>> {
        let n_jobs = if n_jobs < 0 { 0 } else { n_jobs as usize };
        let mut result = ndarray::Array2::zeros((tss.len(), feature_evaluator.size_hint()));
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

    fn many_impl<T>(
        feature_evaluator: &Feature<T>,
        lcs: Vec<PyLightCurve<T>>,
        sorted: Option<bool>,
        check: bool,
        is_t_required: bool,
        fill_value: Option<T>,
        n_jobs: i64,
    ) -> Res<ndarray::Array2<T>>
    where
        T: Float + numpy::Element,
    {
        let tss = lcs
            .iter()
            .map(|(t, m, sigma)| {
                Self::ts_from_numpy(feature_evaluator, t, m, sigma, sorted, check, is_t_required)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Self::eval_many_parallel(feature_evaluator, tss, fill_value, n_jobs)
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

    #[allow(clippy::too_many_arguments)]
    fn many_arrow<'py>(
        &self,
        py: Python<'py>,
        lcs: &Bound<'py, PyAny>,
        fill_value: Option<f64>,
        sorted: Option<bool>,
        check: bool,
        n_jobs: i64,
    ) -> Res<Bound<'py, PyUntypedArray>> {
        // Try __arrow_c_stream__ (ChunkedArray) first, then __arrow_c_array__ (Array)
        let chunked: PyChunkedArray = if let Ok(ca) = lcs.extract::<PyChunkedArray>() {
            ca
        } else if let Ok(arr) = lcs.extract::<pyo3_arrow::PyArray>() {
            PyChunkedArray::from_array_refs(vec![arr.array().clone()])
                .map_err(|e| Exception::ValueError(format!("Failed to convert Arrow array: {e}")))?
        } else {
            return Err(Exception::TypeError(
                "Arrow input must implement __arrow_c_array__ or __arrow_c_stream__".to_string(),
            ));
        };
        let schema = validate_arrow_lcs(&chunked)?;
        let is_t_required = self.is_t_required(sorted);

        match schema.dtype {
            ArrowDtype::F32 => {
                let result = Self::many_arrow_impl(
                    &self.feature_evaluator_f32,
                    &chunked,
                    &schema,
                    sorted,
                    check,
                    is_t_required,
                    fill_value.map(|v| v as f32),
                    n_jobs,
                )?;
                Ok(result.into_pyarray(py).as_untyped().clone())
            }
            ArrowDtype::F64 => {
                let result = Self::many_arrow_impl(
                    &self.feature_evaluator_f64,
                    &chunked,
                    &schema,
                    sorted,
                    check,
                    is_t_required,
                    fill_value,
                    n_jobs,
                )?;
                Ok(result.into_pyarray(py).as_untyped().clone())
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn many_arrow_impl<T: ArrowFloat>(
        feature_evaluator: &Feature<T>,
        chunked: &PyChunkedArray,
        schema: &ArrowLcsSchema,
        sorted: Option<bool>,
        check: bool,
        is_t_required: bool,
        fill_value: Option<T>,
        n_jobs: i64,
    ) -> Res<ndarray::Array2<T>> {
        match schema.list_type {
            ArrowListType::List => Self::many_arrow_chunks::<T, i32>(
                feature_evaluator,
                chunked,
                schema.has_sigma,
                sorted,
                check,
                is_t_required,
                fill_value,
                n_jobs,
            ),
            ArrowListType::LargeList => Self::many_arrow_chunks::<T, i64>(
                feature_evaluator,
                chunked,
                schema.has_sigma,
                sorted,
                check,
                is_t_required,
                fill_value,
                n_jobs,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn many_arrow_chunks<T: ArrowFloat, O: arrow_array::OffsetSizeTrait>(
        feature_evaluator: &Feature<T>,
        chunked: &PyChunkedArray,
        has_sigma: bool,
        sorted: Option<bool>,
        check: bool,
        is_t_required: bool,
        fill_value: Option<T>,
        n_jobs: i64,
    ) -> Res<ndarray::Array2<T>> {
        let tss = chunked
            .chunks()
            .iter()
            .map(|chunk| {
                let list: &arrow_array::GenericListArray<O> = chunk.as_list::<O>();
                // O(1) null checks — nulls are not supported yet
                if list.null_count() > 0 {
                    return Err(Exception::NotImplementedError(
                        "Null entries in the list array are not supported".to_string(),
                    ));
                }
                let struct_arr = list.values().as_struct();
                if struct_arr.null_count() > 0 {
                    return Err(Exception::NotImplementedError(
                        "Null entries in the struct array are not supported".to_string(),
                    ));
                }
                for i in 0..struct_arr.num_columns() {
                    if struct_arr.column(i).null_count() > 0 {
                        return Err(Exception::NotImplementedError(
                            "Null values in data columns are not supported".to_string(),
                        ));
                    }
                }

                let t_vals: &[T] = struct_arr
                    .column(0)
                    .as_primitive::<T::ArrowType>()
                    .values()
                    .as_ref();
                let m_vals: &[T] = struct_arr
                    .column(1)
                    .as_primitive::<T::ArrowType>()
                    .values()
                    .as_ref();
                let sigma_vals: Option<&[T]> = has_sigma.then(|| {
                    struct_arr
                        .column(2)
                        .as_primitive::<T::ArrowType>()
                        .values()
                        .as_ref()
                });
                let offsets = list.value_offsets();
                offsets
                    .iter()
                    .tuple_windows()
                    .map(move |(&start, &end)| {
                        let (start, end) = (start.as_usize(), end.as_usize());
                        Self::ts_from_views(
                            feature_evaluator,
                            ndarray::ArrayView1::from(&t_vals[start..end]),
                            ndarray::ArrayView1::from(&m_vals[start..end]),
                            sigma_vals.map(|s| ndarray::ArrayView1::from(&s[start..end])),
                            sorted,
                            check,
                            is_t_required,
                        )
                    })
                    .collect::<Res<Vec<_>>>()
            })
            .flatten_ok()
            .collect::<Res<Vec<_>>>()?;

        if tss.is_empty() {
            return Err(Exception::ValueError("lcs is empty".to_string()));
        }

        Self::eval_many_parallel(feature_evaluator, tss, fill_value, n_jobs)
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
        fill_value = None,
        sorted = None,
        check = true,
        cast = false,
    ))]
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        t: Bound<'py, PyAny>,
        m: Bound<'py, PyAny>,
        sigma: Option<Bound<'py, PyAny>>,
        fill_value: Option<f64>,
        sorted: Option<bool>,
        check: bool,
        cast: bool,
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
                =sigma;
                cast=cast
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
                =m;
                cast=cast
            )
        }
    }

    #[doc = METHOD_MANY_DOC!()]
    #[pyo3(signature = (lcs, *, fill_value=None, sorted=None, check=true, n_jobs=-1))]
    fn many<'py>(
        &self,
        py: Python<'py>,
        lcs: Bound<'py, PyAny>,
        fill_value: Option<f64>,
        sorted: Option<bool>,
        check: bool,
        n_jobs: i64,
    ) -> Res<Bound<'py, PyUntypedArray>> {
        // Try Arrow path first
        if lcs.hasattr("__arrow_c_array__")? || lcs.hasattr("__arrow_c_stream__")? {
            return self.many_arrow(py, &lcs, fill_value, sorted, check, n_jobs);
        }
        // Fall back to list-of-tuples path
        let lcs: PyLcs<'py> = lcs.extract()?;
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

    /// Used by copy.copy
    fn __copy__(&self) -> Self {
        self.clone()
    }

    /// Used by copy.deepcopy
    fn __deepcopy__(&self, _memo: Bound<PyAny>) -> Self {
        self.clone()
    }
}

macro_rules! impl_pickle_serialisation {
    ($name: ident) => {
        #[pymethods]
        impl $name {
            /// Used by pickle.load / pickle.loads
            fn __setstate__(mut slf: PyRefMut<'_, Self>, state: Bound<PyBytes>) -> Res<()> {
                let (super_rust, self_rust): (PyFeatureEvaluator, Self) = serde_pickle::from_slice(state.as_bytes(), serde_pickle::DeOptions::new())
                    .map_err(|err| {
                        Exception::UnpicklingError(format!(
                            r#"Error happened on the Rust side when deserializing _FeatureEvaluator: "{err}""#
                        ))
                    })?;
                *slf.as_mut() = super_rust;
                *slf = self_rust;
                Ok(())
            }

            /// Used by pickle.dump / pickle.dumps
            fn __getstate__<'py>(slf: PyRef<'py, Self>) -> Res<Bound<'py, PyBytes>> {
                let supr = slf.as_super();
                let vec_bytes = serde_pickle::to_vec(&(supr.deref(), slf.deref()), serde_pickle::SerOptions::new()).map_err(|err| {
                    Exception::PicklingError(format!(
                        r#"Error happened on the Rust side when serializing _FeatureEvaluator: "{err}""#
                    ))
                })?;
                Ok(PyBytes::new(slf.py(), &vec_bytes))
            }
        }
    }
}

#[derive(Serialize, Deserialize)]
#[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
pub struct Extractor {}

impl_pickle_serialisation!(Extractor);

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
                feature_evaluator_f32: FeatureExtractor::new(evals_f32).into(),
                feature_evaluator_f64: FeatureExtractor::new(evals_f64).into(),
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
            FeatureExtractor::<f64, Feature<f64>>::doc().trim_start(),
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
        #[derive(Serialize, Deserialize)]
        #[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
        pub struct $name {}

        impl_stock_transform!($name, $default_transform);

        impl_pickle_serialisation!($name);

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
const N_ALGO_CURVE_FIT_NUTS: usize = {
    #[cfg(feature = "nuts")]
    {
        1 + N_ALGO_CURVE_FIT_CERES / 2 + N_ALGO_CURVE_FIT_GSL / 2
    }
    #[cfg(not(feature = "nuts"))]
    {
        0
    }
};
const N_ALGO_CURVE_FIT: usize = N_ALGO_CURVE_FIT_CERES
    + N_ALGO_CURVE_FIT_GSL
    + N_ALGO_CURVE_FIT_PURE_MCMC
    + N_ALGO_CURVE_FIT_NUTS;

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
    #[cfg(feature = "nuts")]
    "nuts",
    #[cfg(all(
        feature = "nuts",
        any(feature = "ceres-source", feature = "ceres-system")
    ))]
    "nuts-ceres",
    #[cfg(all(feature = "nuts", feature = "gsl"))]
    "nuts-lmsder",
];

macro_const! {
    const FIT_METHOD_MODEL_DOC: &str = r#"model(t, params, *, cast=False)
    Underlying parametric model function

    Parameters
    ----------
    t : np.ndarray of np.float32 or np.float64
        Time moments, can be unsorted
    params : np.ndaarray of np.float32 or np.float64
        Parameters of the model, this array can be longer than actual parameter
        list, the beginning part of the array will be used in this case, see
        Examples section in the class documentation.
    cast : bool, optional
        Cast inputs to np.ndarray of the same dtype

    Returns
    -------
    np.ndarray of np.float32 or np.float64
        Array of model values corresponded to the given time moments
"#;
}

#[derive(FromPyObject)]
pub(crate) enum FitLnPrior {
    #[pyo3(transparent, annotation = "str")]
    Name(String),
    #[pyo3(transparent, annotation = "list[LnPrior]")]
    ListLnPrior1D(Vec<LnPrior1D>),
}

macro_rules! fit_evaluator {
    ($name: ident, $eval: ty, $ib: ty, $transform: expr, $nparam: literal, $ln_prior_by_str: tt, $ln_prior_doc: literal $(,)?) => {
        #[derive(Serialize, Deserialize)]
        #[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
        pub struct $name {}

        impl_pickle_serialisation!($name);

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
                T: Float + numpy::Element,
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

            fn default_nuts_ntune() -> Option<u32> {
                #[cfg(feature = "nuts")]
                {
                    lcf::NutsCurveFit::default_num_tune().into()
                }
                #[cfg(not(feature = "nuts"))]
                {
                    None
                }
            }

            fn default_nuts_ndraws() -> Option<u32> {
                #[cfg(feature = "nuts")]
                {
                    lcf::NutsCurveFit::default_num_draws().into()
                }
                #[cfg(not(feature = "nuts"))]
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
                nuts_ntune = Self::default_nuts_ntune(),
                nuts_ndraws = Self::default_nuts_ndraws(),
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
                // The first Option is for Python's None, the second is for compile-time None from
                // Self::default_nuts_ntune()
                nuts_ntune: Option<Option<u32>>,
                // The first Option is for Python's None, the second is for compile-time None from
                // Self::default_nuts_ndraws()
                nuts_ndraws: Option<Option<u32>>,
                init: Option<Vec<Option<f64>>>,
                bounds: Option<Vec<(Option<f64>, Option<f64>)>>,
                ln_prior: Option<FitLnPrior>,
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

                #[cfg(feature = "nuts")]
                let nuts_ntune_value = nuts_ntune.unwrap_or_else(Self::default_nuts_ntune).expect("logical error: default nuts_ntune is None but nuts is enabled");
                #[cfg(feature = "nuts")]
                let nuts_ndraws_value = nuts_ndraws.unwrap_or_else(Self::default_nuts_ndraws).expect("logical error: default nuts_ndraws is None but nuts is enabled");
                #[cfg(not(feature = "nuts"))]
                if nuts_ntune.flatten().is_some() || nuts_ndraws.flatten().is_some() {
                    return Err(PyValueError::new_err(
                        "Compiled without NUTS support, nuts_ntune and nuts_ndraws are not supported",
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
                        FitLnPrior::Name(s) => match s.as_str() $ln_prior_by_str,
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
                    #[cfg(feature = "nuts")]
                    "nuts" => lcf::NutsCurveFit::new(nuts_ntune_value, nuts_ndraws_value, None).into(),
                    #[cfg(all(feature = "nuts", any(feature = "ceres-source", feature = "ceres-system")))]
                    "nuts-ceres" => lcf::NutsCurveFit::new(nuts_ntune_value, nuts_ndraws_value, Some(ceres_fit)).into(),
                    #[cfg(all(feature = "nuts", feature = "gsl"))]
                    "nuts-lmsder" => lcf::NutsCurveFit::new(nuts_ntune_value, nuts_ndraws_value, Some(lmsder_fit)).into(),
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
                    Some(py_transform) => match py_transform.cast::<PyBool>() {
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
            #[pyo3(signature = (t, params, *, cast=false))]
            fn model<'py>(
                py: Python<'py>,
                t: Bound<'py, PyAny>,
                params: Bound<'py, PyAny>,
                cast: bool
            ) -> Res<Bound<'py, PyUntypedArray>> {
                dtype_dispatch!({
                    |t, params| Ok(Self::model_impl(t, params).into_pyarray(py).as_untyped().clone())
                }(t, !=params; cast=cast))
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
    a number, the loss function is regularized to descriminate outlier
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

#[derive(Serialize, Deserialize)]
#[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
pub struct BeyondNStd {}

impl_stock_transform!(BeyondNStd, StockTransformer::Identity);
impl_pickle_serialisation!(BeyondNStd);

#[pymethods]
impl BeyondNStd {
    #[new]
    #[pyo3(signature = (nstd=lcf::BeyondNStd::<f32>::default_nstd(), *, transform=None))]
    fn __new__(nstd: f32, transform: Option<Bound<PyAny>>) -> Res<PyClassInitializer<Self>> {
        Ok(
            PyClassInitializer::from(PyFeatureEvaluator::with_py_transform(
                lcf::BeyondNStd::new(nstd).into(),
                lcf::BeyondNStd::new(nstd).into(),
                transform,
                Self::DEFAULT_TRANSFORMER,
            )?)
            .add_subclass(Self {}),
        )
    }

    /// Required by pickle.load / pickle.loads
    #[staticmethod]
    fn __getnewargs__() -> (f32,) {
        (lcf::BeyondNStd::<f32>::default_nstd(),)
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

#[derive(Serialize, Deserialize)]
#[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
pub struct Bins {}

impl_pickle_serialisation!(Bins);

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
        for x in features.try_iter()? {
            let py_feature = x?.cast::<PyFeatureEvaluator>()?.borrow();
            eval_f32.add_feature(py_feature.feature_evaluator_f32.clone());
            eval_f64.add_feature(py_feature.feature_evaluator_f64.clone());
        }

        eval_f32.set_window(window);
        eval_f64.set_window(window);

        eval_f32.set_offset(offset);
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
            (PyTuple::empty(py),),
            [
                ("window", lcf::Bins::<f64, Feature<f64>>::default_window()),
                ("offset", lcf::Bins::<f64, Feature<f64>>::default_offset()),
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
            header = lcf::Bins::<f64, Feature<f64>>::doc().trim_start(),
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

#[derive(Serialize, Deserialize)]
#[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
pub struct InterPercentileRange {}

impl_stock_transform!(InterPercentileRange, StockTransformer::Identity);
impl_pickle_serialisation!(InterPercentileRange);

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

#[derive(Serialize, Deserialize)]
#[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
pub struct MagnitudePercentageRatio {}

impl_stock_transform!(MagnitudePercentageRatio, StockTransformer::Identity);
impl_pickle_serialisation!(MagnitudePercentageRatio);

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

#[derive(Serialize, Deserialize)]
#[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
pub struct MedianBufferRangePercentage {}

impl_stock_transform!(MedianBufferRangePercentage, StockTransformer::Identity);
impl_pickle_serialisation!(MedianBufferRangePercentage);

#[pymethods]
impl MedianBufferRangePercentage {
    #[new]
    #[pyo3(signature = (quantile=lcf::MedianBufferRangePercentage::<f32>::default_quantile(), *, transform = None))]
    fn __new__(quantile: f32, transform: Option<Bound<PyAny>>) -> Res<PyClassInitializer<Self>> {
        Ok(
            PyClassInitializer::from(PyFeatureEvaluator::with_py_transform(
                lcf::MedianBufferRangePercentage::new(quantile).into(),
                lcf::MedianBufferRangePercentage::new(quantile).into(),
                transform,
                Self::DEFAULT_TRANSFORMER,
            )?)
            .add_subclass(Self {}),
        )
    }

    /// Required by pickle.load / pickle.loads
    #[staticmethod]
    fn __getnewargs__() -> (f32,) {
        (lcf::MedianBufferRangePercentage::<f32>::default_quantile(),)
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

#[derive(Serialize, Deserialize)]
#[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
pub struct PercentDifferenceMagnitudePercentile {}

impl_stock_transform!(
    PercentDifferenceMagnitudePercentile,
    StockTransformer::ClippedLg
);
impl_pickle_serialisation!(PercentDifferenceMagnitudePercentile);

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

type LcfPeriodogram<T> = lcf::Periodogram<T, Feature<T>>;

#[derive(FromPyObject)]
enum NyquistArgumentOfPeriodogram {
    String(String),
    Float(f32),
}

#[derive(Serialize, Deserialize)]
#[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
pub struct Periodogram {
    eval_f32: LcfPeriodogram<f32>,
    eval_f64: LcfPeriodogram<f64>,
}

impl_pickle_serialisation!(Periodogram);

impl Periodogram {
    fn parse_normalization(normalization: &str) -> PyResult<PeriodogramNormalization> {
        match normalization {
            "psd" => Ok(PeriodogramNormalization::Psd),
            "standard" => Ok(PeriodogramNormalization::Standard),
            "model" => Ok(PeriodogramNormalization::Model),
            "log" => Ok(PeriodogramNormalization::Log),
            _ => Err(PyValueError::new_err(format!(
                "normalization must be one of: 'psd', 'standard', 'model', 'log', got '{normalization}'"
            ))),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn create_evals(
        peaks: Option<usize>,
        resolution: Option<f32>,
        max_freq_factor: Option<f32>,
        nyquist: Option<NyquistArgumentOfPeriodogram>,
        freqs: Option<Bound<PyAny>>,
        fast: Option<bool>,
        features: Option<Bound<PyAny>>,
        normalization: PeriodogramNormalization,
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
            let nyquist_freq: lcf::NyquistFreq = match nyquist {
                NyquistArgumentOfPeriodogram::String(nyquist_type) => match nyquist_type.as_str() {
                    "average" => lcf::AverageNyquistFreq {}.into(),
                    "median" => lcf::MedianNyquistFreq {}.into(),
                    _ => {
                        return Err(PyValueError::new_err(
                            "nyquist must be one of: None, 'average', 'median' or quantile value",
                        ));
                    }
                },
                NyquistArgumentOfPeriodogram::Float(quantile) => {
                    lcf::QuantileNyquistFreq { quantile }.into()
                }
            };
            eval_f32.set_nyquist(nyquist_freq);
            eval_f64.set_nyquist(nyquist_freq);
        }

        let fast = fast.unwrap_or(false);
        if fast {
            #[cfg(feature = "mkl")]
            {
                eval_f32.set_periodogram_algorithm(
                    lcf::PeriodogramPowerFft::<f32, lcf::periodogram::FftwFft<f32>>::new().into(),
                );
                eval_f64.set_periodogram_algorithm(
                    lcf::PeriodogramPowerFft::<f64, lcf::periodogram::FftwFft<f64>>::new().into(),
                );
            }
            #[cfg(not(feature = "mkl"))]
            {
                eval_f32.set_periodogram_algorithm(
                    lcf::PeriodogramPowerFft::<f32, lcf::periodogram::RustFft<f32>>::new().into(),
                );
                eval_f64.set_periodogram_algorithm(
                    lcf::PeriodogramPowerFft::<f64, lcf::periodogram::RustFft<f64>>::new().into(),
                );
            }
        } else {
            eval_f32.set_periodogram_algorithm(lcf::PeriodogramPowerDirect {}.into());
            eval_f64.set_periodogram_algorithm(lcf::PeriodogramPowerDirect {}.into());
        }

        if let Some(freqs) = freqs {
            const STEP_SIZE_TOLLERANCE: f64 = 10.0 * f32::EPSILON as f64;

            // It is more likely for users to give f64 array
            let freqs_f64 = PyArrayLike1::<f64, AllowTypeChange>::extract(freqs.as_borrowed())?;
            let freqs_f64 = freqs_f64.readonly();
            let freqs_f64 = freqs_f64.as_array();
            let size = freqs_f64.len();
            if size < 2 {
                return Err(PyValueError::new_err("freqs must have at least two values"));
            }
            let first_zero = freqs_f64[0].is_zero();
            if fast && !first_zero {
                return Err(PyValueError::new_err(
                    "When Periodogram(freqs=[...], fast=True), freqs[0] must equal 0",
                ));
            }
            let len_is_pow2_p1 = (size - 1).is_power_of_two();
            if fast && !len_is_pow2_p1 {
                return Err(PyValueError::new_err(
                    "When Periodogram(freqs=[...], fast=True), len(freqs) must be a power of two plus one, e.g. 2**k + 1",
                ));
            }
            let step_candidate = freqs_f64[1] - freqs_f64[0];
            let is_linear = freqs_f64.iter().tuple_windows().all(|(x1, x2)| {
                let dx = x2 - x1;
                let rel_diff = f64::abs(dx / step_candidate - 1.0);
                rel_diff < STEP_SIZE_TOLLERANCE
            });
            let freq_grid_f64 = if is_linear {
                if first_zero && len_is_pow2_p1 {
                    let log2_size_m1 = (size - 1).ilog2();
                    FreqGrid::zero_based_pow2(step_candidate, log2_size_m1)
                } else {
                    FreqGrid::linear(freqs_f64[0], step_candidate, size)
                }
            } else if fast {
                return Err(PyValueError::new_err(
                    "When Periodogram(freqs=[...], fast=True), freqs must be a linear grid, like np.linspace(0, max_freq, 2**k + 1)",
                ));
            } else {
                FreqGrid::from_array(&freqs_f64)
            };

            let freq_grid_f32 = match &freq_grid_f64 {
                FreqGrid::Arbitrary(_) => {
                    let freqs_f32 =
                        PyArrayLike1::<f32, AllowTypeChange>::extract(freqs.as_borrowed())?;
                    let freqs_f32 = freqs_f32.readonly();
                    let freqs_f32 = freqs_f32.as_array();
                    FreqGrid::from_array(&freqs_f32)
                }
                FreqGrid::Linear(_) => {
                    FreqGrid::linear(freqs_f64[0] as f32, step_candidate as f32, size)
                }
                FreqGrid::ZeroBasedPow2(_) => {
                    FreqGrid::zero_based_pow2(step_candidate as f32, (size - 1).ilog2())
                }
                _ => {
                    panic!("This FreqGrid is not implemented yet")
                }
            };

            eval_f32.set_freq_grid(freq_grid_f32);
            eval_f64.set_freq_grid(freq_grid_f64);
        }

        if let Some(features) = features {
            for x in features.try_iter()? {
                let py_feature = x?.cast::<PyFeatureEvaluator>()?.borrow();
                eval_f32.add_feature(py_feature.feature_evaluator_f32.clone());
                eval_f64.add_feature(py_feature.feature_evaluator_f64.clone());
            }
        }

        eval_f32.set_normalization(normalization);
        eval_f64.set_normalization(normalization);

        Ok((eval_f32, eval_f64))
    }

    fn power_impl<'py, T>(
        eval: &lcf::Periodogram<T, Feature<T>>,
        py: Python<'py>,
        t: Arr<T>,
        m: Arr<T>,
    ) -> Res<Bound<'py, PyUntypedArray>>
    where
        T: Float + numpy::Element,
    {
        let t: DataSample<_> = t.as_array().into();
        let m: DataSample<_> = m.as_array().into();
        let mut ts = TimeSeries::new_without_weight(t, m);
        let power = eval.power(&mut ts).map_err(lcf::EvaluatorError::from)?;
        let power = PyArray1::from_vec(py, power);
        Ok(power.as_untyped().clone())
    }

    fn freq_power_impl<'py, T>(
        eval: &lcf::Periodogram<T, Feature<T>>,
        py: Python<'py>,
        t: Arr<T>,
        m: Arr<T>,
    ) -> Res<(Bound<'py, PyUntypedArray>, Bound<'py, PyUntypedArray>)>
    where
        T: Float + numpy::Element,
    {
        let t: DataSample<_> = t.as_array().into();
        let m: DataSample<_> = m.as_array().into();
        let mut ts = TimeSeries::new_without_weight(t, m);
        let (freq, power) = eval
            .freq_power(&mut ts)
            .map_err(lcf::EvaluatorError::from)?;
        let freq = PyArray1::from_vec(py, freq);
        let power = PyArray1::from_vec(py, power);
        Ok((freq.as_untyped().clone(), power.as_untyped().clone()))
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
        nyquist = NyquistArgumentOfPeriodogram::String(String::from("average")),
        freqs = None,
        fast = true,
        features = None,
        normalization = "psd",
        transform = None,
    ))]
    fn __new__(
        peaks: Option<usize>,
        resolution: Option<f32>,
        max_freq_factor: Option<f32>,
        nyquist: Option<NyquistArgumentOfPeriodogram>,
        freqs: Option<Bound<PyAny>>,
        fast: Option<bool>,
        features: Option<Bound<PyAny>>,
        normalization: &str,
        transform: Option<Bound<PyAny>>,
    ) -> PyResult<(Self, PyFeatureEvaluator)> {
        if transform.is_some() {
            return Err(PyNotImplementedError::new_err(
                "transform is not supported by Periodogram, peak-related features are not transformed, but you still may apply transformation for the underlying features",
            ));
        }
        let normalization = Self::parse_normalization(normalization)?;
        let (eval_f32, eval_f64) = Self::create_evals(
            peaks,
            resolution,
            max_freq_factor,
            nyquist,
            freqs,
            fast,
            features,
            normalization,
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

    /// Periodogram values
    #[pyo3(signature = (t, m, *, cast=false))]
    fn power<'py>(
        &self,
        py: Python<'py>,
        t: Bound<PyAny>,
        m: Bound<PyAny>,
        cast: bool,
    ) -> Res<Bound<'py, PyUntypedArray>> {
        dtype_dispatch!(
            |t, m| Self::power_impl(&self.eval_f32, py, t, m),
            |t, m| Self::power_impl(&self.eval_f64, py, t, m),
            t,
            =m;
            cast=cast
        )
    }

    /// Angular frequencies and periodogram values
    #[pyo3(signature = (t, m, *, cast=false))]
    fn freq_power<'py>(
        &self,
        py: Python<'py>,
        t: Bound<PyAny>,
        m: Bound<PyAny>,
        cast: bool,
    ) -> Res<(Bound<'py, PyUntypedArray>, Bound<'py, PyUntypedArray>)> {
        dtype_dispatch!(
            |t, m| Self::freq_power_impl(&self.eval_f32, py, t, m),
            |t, m| Self::freq_power_impl(&self.eval_f64, py, t, m),
            t,
            =m;
            cast=cast
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
freqs : array-like or None, optional
    Explicid and fixed frequency grid (angular frequency, radians/time unit).
    If given, `resolution`, `max_freq_factor` and `nyquist` are being
    ignored.
    For `fast=True` the only supported type of the grid is
    np.linspace(0.0, max_freq, 2**k+1), where k is an integer.
    For `fast=False` any grid is accepted, but linear grids, like
    np.linspace(min_freq, max_freq, n), apply some computational
    optimisations.
fast : bool or None, optional
    Use "Fast" (approximate and FFT-based) or direct periodogram algorithm,
    default is {default_fast}
features : iterable or None, optional
    Features to extract from periodogram considering it as a time-series,
    default is None which means no additional features
    Features to extract from periodogram considering it as a time-series
normalization : str, optional
    Normalization of the periodogram power. Affects `power()`,
    `freq_power()`, and feature extraction via `__call__()`.
    Let P be the raw power and n the number of observations.
    Must be one of:
     - 'psd': Raw power P, unnormalized (default). Consistent with
       scipy.signal.lombscargle(normalize=False) on variance-normalized
       data, but differs from astropy's 'psd' convention
     - 'standard': P_std = P * 2 / (n - 1), values in [0, 1].
       Matches astropy's 'standard' normalization
     - 'model': P_std / (1 - P_std), values in [0, inf).
       Matches astropy's 'model' normalization
     - 'log': -ln(1 - P_std), values in [0, inf).
       Matches astropy's 'log' normalization
    Default is 'psd'
transform : None, optional
    Not supported for Periodogram, peaks are not transformed, but you still
    may apply transformation for the underlying features with thier
    constructors

{common}
freq_power(t, m, *, cast=False)
    Get periodogram as a pair of frequencies and power values

    Parameters
    ----------
    t : np.ndarray of np.float32 or np.float64
        Time array
    m : np.ndarray of np.float32 or np.float64
        Magnitude (flux) array
    cast : bool, optional
        Cast inputs to np.ndarray objects of the same dtype

    Returns
    -------
    freq : np.ndarray of np.float32 or np.float64
        Frequency grid
    power : np.ndarray of np.float32 or np.float64
        Periodogram power

power(t, m, *, cast=False)
    Get periodogram power

    Parameters
    ----------
    t : np.ndarray of np.float32 or np.float64
        Time array
    m : np.ndarray of np.float32 or np.float64
        Magnitude (flux) array
    cast : bool, optional
        Cast inputs to np.ndarray objects of the same dtype

    Returns
    -------
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

#[derive(Serialize, Deserialize)]
#[pyclass(extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
pub struct OtsuSplit {}

impl_pickle_serialisation!(OtsuSplit);

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
        T: Float + numpy::Element,
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
#[derive(Serialize, Deserialize)]
#[pyclass(name = "JSONDeserializedFeature", extends = PyFeatureEvaluator, module="light_curve.light_curve_ext")]
pub struct JsonDeserializedFeature {}

impl_pickle_serialisation!(JsonDeserializedFeature);

#[pymethods]
impl JsonDeserializedFeature {
    #[new]
    #[pyo3(text_signature = "(json_string)")]
    fn __new__(s: String) -> Res<(Self, PyFeatureEvaluator)> {
        let feature_evaluator_f32: Feature<f32> = serde_json::from_str(&s).map_err(|err| {
            Exception::ValueError(format!("Cannot deserialize feature from JSON: {err}"))
        })?;
        let feature_evaluator_f64: Feature<f64> = serde_json::from_str(&s).map_err(|err| {
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
