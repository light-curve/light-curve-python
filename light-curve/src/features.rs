use crate::arrow_input::{
    ArrowBandType, ArrowDtype, ArrowFloat, ArrowLcsSchema, ArrowListType, PyArrowFields,
    validate_arrow_lcs,
};
use crate::check::{check_finite, check_no_nans, is_sorted};
use crate::cont_array::ContCowArray;
use crate::errors::{Exception, Res};
use crate::ln_prior::LnPrior1D;
use crate::np_array::Arr;
use crate::transform::{StockTransformer, parse_transform};

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::ArrowPrimitiveType;
use const_format::formatcp;
use conv::ConvUtil;
use itertools::Itertools;
use light_curve_feature::{
    self as lcf, DataSample, MultiColorEvaluator, PassbandTrait,
    periodogram::{FreqGrid, PeriodogramNormalization},
    prelude::*,
};
use macro_const::macro_const;
use ndarray::IntoNdProducer;
use num_traits::{AsPrimitive, Zero};
use numpy::prelude::*;
use numpy::{AllowTypeChange, PyArray1, PyArrayLike1, PyUntypedArray};
use once_cell::sync::OnceCell;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyModule, PyTuple};
use pyo3_arrow::PyChunkedArray;
use rayon::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::convert::TryInto;
use std::fmt;
use std::ops::Deref;
use std::ops::Range;
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

type PyMultibandLcs<'py> = Vec<(
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Option<Bound<'py, PyAny>>,
    Bound<'py, PyAny>,
)>;

type OwnedMultibandLc<T> = (
    ndarray::Array1<T>,
    ndarray::Array1<T>,
    Option<ndarray::Array1<T>>,
    Vec<usize>,
);

const ATTRIBUTES_DOC: &str = r#"Attributes
----------
names : list of str
    Feature names
descriptions : list of str
    Feature descriptions
bands : numpy.ndarray of str or None
    Passband names for multiband mode, or None for single-band mode"#;

macro_const! {
    const METHOD_CALL_DOC: &str = r#"__call__(self, t, m, sigma=None, band=None, *, fill_value=None, sorted=None, check=True, cast=False)
    Extract features and return them as a numpy array

    Parameters
    ----------
    t : numpy.ndarray of float32 or float64
        Time moments

    m : numpy.ndarray
        Signal in magnitude or fluxes. Refer to the feature description to
        decide which would work better in your case

    sigma : numpy.ndarray, default None
        Observation error, if None it is assumed to be unity

    band : numpy.ndarray of str, optional
        Passband label for each observation. Required in multiband mode
        (when the feature was constructed with ``bands=``), ignored
        in single-band mode.

    fill_value : float or None, default None
        Value to fill invalid feature values, for example if count of
        observations is not enough to find a proper value.
        None causes exception for invalid features

    sorted : bool or None, default None
        Specifies if input array are sorted by time moments.
        True is for certainly sorted, False is for unsorted.
        If None is specified than sorting is checked and an exception is
        raised for unsorted t

    check : bool, default True
        Check all input arrays for NaNs, t and m for infinite values

    cast : bool, default False
        Allows non-numpy input and casting of arrays to a common dtype.
        If False, inputs must be np.ndarray instances with matched dtypes.
        Casting provides more flexibility with input types at the cost of
        performance.

    Returns
    -------
    ndarray of float32 or float64
        Extracted feature array"#;
}

macro_const! {
    const METHOD_MANY_DOC: &str = r#"Extract features from multiple light curves in parallel

    It is a parallel executed equivalent of

    >>> def many(self, lcs, *, fill_value=None, sorted=None, check=True):
    ...     return np.stack([self(*lc, fill_value=fill_value, sorted=sorted,
    ...                          check=check, cast=False) for lc in lcs])

    Parameters
    ----------
    lcs : list of (t, m, sigma) or Arrow array
        Either a list of light curves packed into three-tuples (all numpy.ndarray
        of the same dtype), or an Arrow array/chunked array of type
        List<Struct<...>> where the selected fields share the same float dtype
        (float32 or float64). Arrow input is auto-detected via the
        ``__arrow_c_array__`` / ``__arrow_c_stream__`` protocol and enables
        zero-copy data access from pyarrow, polars, and other Arrow-compatible
        libraries.
        For multiband features: a list of four-tuples ``(t, m, sigma, band)``
        where ``band`` is an array of passband labels.

    arrow_fields : dict
        Required when lcs is an Arrow array. Maps roles to struct field names
        or zero-based indices, e.g.
        ``{"t": "time", "m": "flux", "sigma": "fluxerr", "band": "passband"}``.
        Keys ``"t"`` and ``"m"`` are required; ``"sigma"`` and ``"band"`` are
        optional. ``"band"`` is required for multiband features and must refer
        to a Utf8 or LargeUtf8 column. Ignored for non-Arrow input.

    fill_value : float or None, default None
        Fill invalid values by this or raise an exception if None

    sorted : bool or None, default None
        Specifies if input array are sorted by time moments, see ``__call__``
        documentation for details

    check : bool, default True
        Check all input arrays for NaNs, t and m for infinite values

    n_jobs : int, default -1
        Number of tasks to run in parallel. -1 means run as many jobs as CPU
        count. See rayon rust crate documentation for details"#;
}

const METHODS_DOC: &str = r#"Methods
-------
__call__(self, t, m, sigma=None, band=None, *, fill_value=None, sorted=None, check=True, cast=False)
    Extract features and return them as a numpy array
many(self, lcs, *, fill_value=None, sorted=None, check=True, cast=False, n_jobs=-1)
    Extract features from multiple light curves in parallel"#;

const COMMON_FEATURE_DOC: &str = formatcp!("\n{}\n\n{}\n", ATTRIBUTES_DOC, METHODS_DOC);

/// Prepare upstream Rust doc strings for MkDocs/Arithmatex rendering.
///
/// Arithmatex uses a BlockProcessor that receives one Markdown block at a time
/// (text between blank lines). A `$$...equation...\n$$` block must therefore
/// have NO blank lines inside it, but MUST have blank lines before the opening
/// `$$` and after the closing `$$`.
///
/// This function:
/// 1. Adds a blank line before an opening `$$` when the previous line was not
///    already blank.
/// 2. Adds a blank line after a closing `$$` when the next line is not already
///    blank.
/// 3. Trims leading whitespace (replaces the old `.trim_start()` call).
fn prepare_upstream_doc(s: &str) -> String {
    // Split into lines and skip leading blank lines
    let raw_lines: Vec<&str> = s.lines().collect();
    let start = raw_lines
        .iter()
        .position(|l| !l.trim().is_empty())
        .unwrap_or(0);
    let raw_lines = &raw_lines[start..];

    // Dedent: compute the minimum leading-space count across all non-empty lines.
    // This prevents indented `$$` from being parsed as a Markdown code block.
    let indent = raw_lines
        .iter()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.len() - l.trim_start().len())
        .min()
        .unwrap_or(0);

    // Build a vec of dedented lines for processing
    let lines: Vec<&str> = raw_lines
        .iter()
        .map(|l| {
            if l.trim().is_empty() {
                ""
            } else {
                &l[indent..]
            }
        })
        .collect();

    let mut result = String::with_capacity(s.len() + 64);
    let mut in_math = false;

    for (i, &line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        if trimmed == "$$" {
            if !in_math {
                // Opening $$: insert blank line before if previous line is non-blank.
                let prev_blank = i == 0 || lines[i - 1].trim().is_empty();
                if !prev_blank {
                    result.push('\n');
                }
                in_math = true;
            } else {
                // Closing $$: output first, then insert blank line after if the
                // next line is non-blank (keeps math in its own block for Arithmatex).
                in_math = false;
                result.push_str(line);
                result.push('\n');
                let next_blank = i + 1 >= lines.len() || lines[i + 1].trim().is_empty();
                if !next_blank {
                    result.push('\n');
                }
                continue;
            }
        }

        result.push_str(line);
        result.push('\n');
    }

    // Preserve original trailing-newline behaviour
    let trimmed_s = s.trim_end();
    if !trimmed_s.ends_with('\n') && result.ends_with('\n') {
        result.pop();
    }
    result
}

const BANDS_PARAMETER_DOC: &str = r#"bands : list of str or None, optional
    Passband names for multiband mode. If provided, the feature is evaluated
    independently per passband and the outputs are concatenated in passband
    order. If None (default), single-band mode is used."#;

fn transform_parameter_doc(default: StockTransformer) -> String {
    let default_name: &str = default.into();
    let variants = StockTransformer::all_variants().format_with("\n     - ", |variant, fmt| {
        let name: &str = variant.into();
        let doc = variant.doc().trim();
        fmt(&format_args!("``'{name}'`` - {doc}"))
    });
    format!(
        r#"transform : str or bool or None, default None
    Transformer to apply to the feature values. If str, must be one of:

     - ``'default'`` - use default transformer for the feature, same as giving
       True. The default for this feature is ``'{default_name}'``
     - {variants}

    If bool, True uses the default transformer, False disables it.
    If None, no transformation is applied (default)"#,
    )
}

type PyLightCurve<'a, T> = (Arr<'a, T>, Arr<'a, T>, Option<Arr<'a, T>>);

/// A passband label: either a string name (e.g. "g", "r") or an integer ID.
///
/// Unifies [`lcf::StringPassband`] and [`lcf::LabeledPassband<i64>`] under a single type
/// that satisfies [`PassbandTrait`], so [`lcf::MultiColorFeature`] can be parameterised with
/// one concrete passband type regardless of how the user supplied the band labels.
///
/// Serde uses the untagged representation: string bands serialise as bare JSON strings
/// (e.g. `"g"`), integer bands as `{"label": 0}`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub(crate) enum Passband {
    String(lcf::StringPassband),
    Integer(lcf::LabeledPassband<i64>),
}

impl PassbandTrait for Passband {
    fn name(&self) -> &str {
        match self {
            Passband::String(p) => p.name(),
            Passband::Integer(p) => p.name(),
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "mode")]
#[allow(clippy::large_enum_variant)]
pub enum FeatureEvalMode {
    SingleBand {
        feature_evaluator_f32: Feature<f32>,
        feature_evaluator_f64: Feature<f64>,
    },
    MultiBand {
        /// Bands in user-specified order — exposed as the `.bands` Python property.
        bands: Vec<Passband>,
        feature_evaluator_f32: Box<lcf::MultiColorFeature<Passband, f32>>,
        feature_evaluator_f64: Box<lcf::MultiColorFeature<Passband, f64>>,
        /// Bands sorted by `Passband::Ord` for use with `MultiColorTimeSeries`.
        #[serde(skip)]
        sorted_bands: Vec<Passband>,
        #[serde(skip)]
        band_input: BandInput,
    },
    /// Mixed single-band + multiband features, combined by Extractor.
    ///
    /// At call time, single-band and multiband sub-evaluators run independently;
    /// their outputs are interleaved according to `sb_mask` (true = single-band slot).
    Mixed {
        /// Bands in user-specified order — exposed as the `.bands` Python property.
        bands: Vec<Passband>,
        sb_f32: Feature<f32>,
        sb_f64: Feature<f64>,
        mc_f32: Box<lcf::MultiColorFeature<Passband, f32>>,
        mc_f64: Box<lcf::MultiColorFeature<Passband, f64>>,
        /// Length = total output size. `true` = this output slot comes from the
        /// single-band sub-evaluator, `false` = from the multiband one.
        sb_mask: Vec<bool>,
        /// Bands sorted by `Passband::Ord` for use with `MultiColorTimeSeries`.
        #[serde(skip)]
        sorted_bands: Vec<Passband>,
        #[serde(skip)]
        band_input: BandInput,
    },
}

impl Clone for FeatureEvalMode {
    fn clone(&self) -> Self {
        match self {
            Self::SingleBand {
                feature_evaluator_f32,
                feature_evaluator_f64,
            } => Self::SingleBand {
                feature_evaluator_f32: feature_evaluator_f32.clone(),
                feature_evaluator_f64: feature_evaluator_f64.clone(),
            },
            Self::MultiBand {
                bands,
                feature_evaluator_f32,
                feature_evaluator_f64,
                ..
            } => {
                let bands = bands.clone();
                let (sorted_bands, band_input) = make_sorted_bands_and_input(&bands);
                Self::MultiBand {
                    band_input,
                    sorted_bands,
                    bands,
                    feature_evaluator_f32: feature_evaluator_f32.clone(),
                    feature_evaluator_f64: feature_evaluator_f64.clone(),
                }
            }
            Self::Mixed {
                bands,
                sb_f32,
                sb_f64,
                mc_f32,
                mc_f64,
                sb_mask,
                ..
            } => {
                let bands = bands.clone();
                let (sorted_bands, band_input) = make_sorted_bands_and_input(&bands);
                Self::Mixed {
                    band_input,
                    sorted_bands,
                    bands,
                    sb_f32: sb_f32.clone(),
                    sb_f64: sb_f64.clone(),
                    mc_f32: mc_f32.clone(),
                    mc_f64: mc_f64.clone(),
                    sb_mask: sb_mask.clone(),
                }
            }
        }
    }
}

/// Base class for all feature extractors.
///
/// Call signature:
/// ``extractor(t, m, sigma=None, band=None, *, fill_value=None, sorted=None, check=True, cast=False)``
#[derive(Serialize, Deserialize)]
#[pyclass(
    subclass,
    name = "_FeatureEvaluator",
    module = "light_curve.light_curve_ext",
    from_py_object
)]
pub struct PyFeatureEvaluator {
    mode: FeatureEvalMode,
}

impl Clone for PyFeatureEvaluator {
    fn clone(&self) -> Self {
        Self {
            mode: self.mode.clone(),
        }
    }
}

/// Extract integer bands from a 1-D numpy integer array.
fn extract_int_bands_numpy(arr: &Bound<'_, PyUntypedArray>) -> Res<Vec<Passband>> {
    let i64_arr = arr
        .as_any()
        .extract::<PyArrayLike1<i64, AllowTypeChange>>()
        .map_err(|e| Exception::TypeError(format!("band array to int64 failed: {e}")))?;
    let int_vals = i64_arr.as_array();
    let mut seen = BTreeSet::new();
    for &v in int_vals.iter() {
        if !seen.insert(v) {
            return Err(Exception::ValueError(format!(
                "bands must be unique; duplicate: {v}"
            )));
        }
    }
    Ok(int_vals
        .iter()
        .map(|&v| Passband::Integer(lcf::LabeledPassband::new(v)))
        .collect())
}

/// Extract a Python array-like into a `Vec<Passband>`.
///
/// Integer numpy arrays and Python lists/tuples of integers produce `Passband::Integer` entries;
/// string numpy arrays and lists of str produce `Passband::String` entries.
fn parse_bands(bands_py: &Bound<PyAny>) -> Res<Vec<Passband>> {
    // Fast path for numpy arrays: dispatch on dtype kind.
    if let Ok(arr) = bands_py.cast::<PyUntypedArray>()
        && arr.ndim() == 1
        && arr.is_c_contiguous()
    {
        let kind = arr.dtype().kind();
        match kind {
            b'i' | b'u' => return extract_int_bands_numpy(arr),
            b'S' | b'a' | b'U' | b'O' => {} // fall through to generic iteration
            _ => {
                return Err(Exception::TypeError(format!(
                    "bands array has unsupported dtype kind '{}'; \
                        expected integer (i/u) or string (S/U/O) dtype",
                    kind as char
                )));
            }
        }
    }

    // Generic iteration: detect mode from the first element.
    let mut int_vals: Vec<i64> = Vec::new();
    let mut str_names: Vec<String> = Vec::new();
    let mut mode: Option<bool> = None; // Some(true) = integer, Some(false) = string

    for item_res in bands_py.try_iter()? {
        let item = item_res?;
        match mode {
            None => {
                if let Ok(v) = item.extract::<i64>() {
                    mode = Some(true);
                    int_vals.push(v);
                } else if let Ok(s) = item.extract::<String>() {
                    mode = Some(false);
                    str_names.push(s);
                } else {
                    return Err(Exception::TypeError(
                        "bands elements must be integers or strings".to_string(),
                    ));
                }
            }
            Some(true) => int_vals.push(item.extract::<i64>()?),
            Some(false) => str_names.push(item.extract::<String>()?),
        }
    }

    match mode {
        None => Err(Exception::ValueError("bands must not be empty".to_string())),
        Some(true) => {
            let mut seen = BTreeSet::new();
            for &v in &int_vals {
                if !seen.insert(v) {
                    return Err(Exception::ValueError(format!(
                        "bands must be unique; duplicate: {v}"
                    )));
                }
            }
            Ok(int_vals
                .into_iter()
                .map(|v| Passband::Integer(lcf::LabeledPassband::new(v)))
                .collect())
        }
        Some(false) => {
            let mut seen = BTreeSet::new();
            str_names
                .iter()
                .map(|s| {
                    if !seen.insert(s.clone()) {
                        return Err(Exception::ValueError(format!(
                            "bands must be unique; duplicate: {s}"
                        )));
                    }
                    Ok(Passband::String(lcf::StringPassband::from(s.as_str())))
                })
                .collect()
        }
    }
}

/// Pre-built per-dtype lookup tables for the fast numpy band-array path.
///
/// The number of configured bands is small (typically ≤ 10), so a linear scan over a
/// `Vec` of keys is faster than hashing each observation's bytes: it avoids the SipHash
/// cost entirely and the first key comparison is a cheap length check.
///
/// Keys not longer than 16 bytes additionally get a "packed" table: the zero-padded key
/// bytes reinterpreted as a little-endian `u128`. numpy's typed-string buffers are
/// zero-padded the same way, so a whole null-padded item compares equal to its key with
/// a single integer comparison — no `memcmp` call, no padding-strip scan (profiling
/// showed the per-observation `memcmp` calls dominating the band-parsing cost).
#[derive(Default)]
pub(crate) struct BandLookup {
    /// Band-name bytes as Rust stores them (UTF-8 by construction) → index.
    /// Matched verbatim against S/a-dtype buffers and the Python-iteration fallback;
    /// the comparison itself is encoding-agnostic byte equality.
    raw: Vec<(Box<[u8]>, usize)>,
    /// Unpadded UCS-4 LE bytes → index. Used for U-dtype (no decoding needed).
    ucs4: Vec<(Box<[u8]>, usize)>,
    /// Packed variant of `raw`; `None` if some band name exceeds 16 bytes.
    raw_packed: Option<Vec<(u128, usize)>>,
    /// Packed variant of `ucs4`; `None` if some band name exceeds 4 UCS-4 codepoints.
    ucs4_packed: Option<Vec<(u128, usize)>>,
}

impl BandLookup {
    fn get_raw(&self, key: &[u8]) -> Option<usize> {
        self.raw
            .iter()
            .find_map(|(k, i)| (k.as_ref() == key).then_some(*i))
    }

    fn get_ucs4(&self, key: &[u8]) -> Option<usize> {
        self.ucs4
            .iter()
            .find_map(|(k, i)| (k.as_ref() == key).then_some(*i))
    }
}

/// Zero-pad `bytes` to 16 and reinterpret as a little-endian `u128`;
/// `None` if it doesn't fit.
fn pack_key(bytes: &[u8]) -> Option<u128> {
    (bytes.len() <= 16).then(|| {
        let mut buf = [0u8; 16];
        buf[..bytes.len()].copy_from_slice(bytes);
        u128::from_le_bytes(buf)
    })
}

fn build_band_lookup(bands: &[Passband]) -> BandLookup {
    let raw: Vec<(Box<[u8]>, usize)> = bands
        .iter()
        .enumerate()
        .map(|(i, b)| (b.name().as_bytes().to_vec().into_boxed_slice(), i))
        .collect();
    let ucs4: Vec<(Box<[u8]>, usize)> = bands
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let key: Box<[u8]> = b
                .name()
                .chars()
                .flat_map(|c| (c as u32).to_le_bytes())
                .collect::<Vec<u8>>()
                .into_boxed_slice();
            (key, i)
        })
        .collect();
    let pack_table = |table: &[(Box<[u8]>, usize)]| {
        table
            .iter()
            .map(|(k, i)| pack_key(k).map(|p| (p, *i)))
            .collect::<Option<Vec<_>>>()
    };
    BandLookup {
        raw_packed: pack_table(&raw),
        ucs4_packed: pack_table(&ucs4),
        raw,
        ucs4,
    }
}

/// Maps i64 band IDs → indices into `sorted_bands`. The IDs are in the same order as
/// `sorted_bands` (sorted numerically, i.e. `LabeledPassband<i64>::Ord` = `i64::Ord`).
///
/// `Range` is used when the IDs form a contiguous integer sequence — lookup is O(1) via
/// subtraction. Otherwise `Entries` holds `(id, index)` pairs for a linear scan.
pub(crate) enum IntBandLookup {
    Range(Range<i64>),
    Entries(Vec<(i64, usize)>),
}

impl IntBandLookup {
    fn build(sorted_values: &[i64]) -> Self {
        if let Some(&first) = sorted_values.first() {
            let is_contiguous = sorted_values
                .iter()
                .enumerate()
                .all(|(i, &v)| first.checked_add(i as i64) == Some(v));
            if is_contiguous {
                return Self::Range(first..first + sorted_values.len() as i64);
            }
        }
        Self::Entries(
            sorted_values
                .iter()
                .copied()
                .enumerate()
                .map(|(i, v)| (v, i))
                .collect(),
        )
    }

    pub(crate) fn get(&self, value: i64) -> Option<usize> {
        match self {
            Self::Range(r) => r
                .contains(&value)
                .then(|| usize::try_from(value - r.start).ok())
                .flatten(),
            Self::Entries(entries) => entries.iter().find_map(|&(v, i)| (v == value).then_some(i)),
        }
    }

    /// Cast the k-element lookup table to native type `T` once before an O(N) loop.
    /// Values that don't fit in `T` are dropped (they can never match a `T` value).
    pub(crate) fn native_lookup<T>(&self) -> IntNativeLookup<T>
    where
        T: TryFrom<i64> + Copy,
    {
        match self {
            Self::Range(r) => match (T::try_from(r.start), T::try_from(r.end)) {
                (Ok(start), Ok(end)) => IntNativeLookup::Range(start..end),
                _ => IntNativeLookup::Entries(
                    r.clone()
                        .enumerate()
                        .filter_map(|(i, v)| T::try_from(v).ok().map(|vv| (vv, i)))
                        .collect(),
                ),
            },
            Self::Entries(entries) => IntNativeLookup::Entries(
                entries
                    .iter()
                    .filter_map(|&(v, i)| T::try_from(v).ok().map(|vv| (vv, i)))
                    .collect(),
            ),
        }
    }

    pub(crate) fn sorted_values_display(&self) -> String {
        match self {
            Self::Range(r) => r
                .clone()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", "),
            Self::Entries(entries) => entries
                .iter()
                .map(|(v, _)| v.to_string())
                .collect::<Vec<_>>()
                .join(", "),
        }
    }
}

pub(crate) enum IntNativeLookup<T> {
    Range(Range<T>),
    Entries(Vec<(T, usize)>),
}

impl<T> IntNativeLookup<T>
where
    T: Copy + PartialOrd + num_traits::CheckedSub,
    usize: TryFrom<T>,
{
    fn get(&self, value: T) -> Option<usize> {
        match self {
            Self::Range(r) => r
                .contains(&value)
                .then(|| value.checked_sub(&r.start))
                .flatten()
                .and_then(|d| usize::try_from(d).ok()),
            Self::Entries(entries) => entries
                .iter()
                .find_map(|&(bv, bi)| (bv == value).then_some(bi)),
        }
    }
}

/// How to parse the band array at call time.
pub(crate) enum BandInput {
    String(BandLookup),
    Integer(IntBandLookup),
}

impl Default for BandInput {
    fn default() -> Self {
        BandInput::String(BandLookup::default())
    }
}

/// Build sorted bands and the matching `BandInput` from user-specified bands.
///
/// Sorts `bands` by `Passband::Ord`: string bands lex, integer bands numeric.
/// Determines `BandInput` variant from the first element (all bands in one feature
/// are always the same variant — either all string or all integer).
fn make_sorted_bands_and_input(bands: &[Passband]) -> (Vec<Passband>, BandInput) {
    let mut sorted = bands.to_vec();
    sorted.sort();
    let band_input = match sorted.first() {
        None | Some(Passband::String(_)) => BandInput::String(build_band_lookup(&sorted)),
        Some(Passband::Integer(_)) => {
            let sorted_ints: Vec<i64> = sorted
                .iter()
                .map(|p| match p {
                    Passband::Integer(lp) => lp.label,
                    Passband::String(_) => unreachable!(),
                })
                .collect();
            BandInput::Integer(IntBandLookup::build(&sorted_ints))
        }
    };
    (sorted, band_input)
}

/// Look up a single string band name in the string lookup table (used by arrow path).
fn lookup_str_band(lookup: &BandLookup, sorted_bands: &[Passband], s: &str) -> Res<usize> {
    lookup.get_raw(s.as_bytes()).ok_or_else(|| {
        Exception::ValueError(format!(
            "unknown passband '{}'; configured bands are: {}",
            s,
            sorted_bands
                .iter()
                .map(|b| b.name())
                .collect::<Vec<_>>()
                .join(", ")
        ))
    })
}

/// Fast band-array parsing. Dispatches on `BandInput` to use integer or string lookup.
/// Falls back to Python iteration for non-numpy inputs (object arrays, lists, etc.).
/// Returns per-observation indices into `sorted_bands`.
fn band_array_to_indices(
    band_py: &Bound<PyAny>,
    sorted_bands: &[Passband],
    band_input: &BandInput,
) -> Res<Vec<usize>> {
    match band_input {
        BandInput::String(lookup) => {
            if let Some(indices) = try_band_view_lookup(band_py, lookup)? {
                return Ok(indices);
            }
            band_py
                .try_iter()?
                .map(|item| {
                    let s = item?.extract::<String>()?;
                    lookup.get_raw(s.as_bytes()).ok_or_else(|| {
                        Exception::ValueError(format!(
                            "unknown passband '{}'; configured bands are: {}",
                            s,
                            sorted_bands
                                .iter()
                                .map(|b| b.name())
                                .collect::<Vec<_>>()
                                .join(", ")
                        ))
                    })
                })
                .collect()
        }
        BandInput::Integer(lookup) => {
            if let Some(indices) = try_int_band_view_lookup(band_py, lookup)? {
                return Ok(indices);
            }
            band_py
                .try_iter()?
                .map(|item| {
                    let v = item?.extract::<i64>()?;
                    lookup.get(v).ok_or_else(|| {
                        Exception::ValueError(format!(
                            "unknown passband {v}; configured bands are: {}",
                            lookup.sorted_values_display()
                        ))
                    })
                })
                .collect()
        }
    }
}

/// Iterate a typed `PyArray1<T>` (borrowed from `band_py`), widening each element to i64 for lookup.
fn int_band_view_lookup_typed<T>(band_py: &Bound<PyAny>, lookup: &IntBandLookup) -> Res<Vec<usize>>
where
    T: numpy::Element + AsPrimitive<i64> + fmt::Display,
{
    let typed = band_py
        .cast::<PyArray1<T>>()
        .map_err(|e| Exception::TypeError(format!("int band downcast failed: {e}")))?;
    let ro = typed.readonly();
    let values = ro
        .as_slice()
        .map_err(|_| Exception::ValueError("band array is not contiguous".to_string()))?;
    values
        .iter()
        .map(|&raw| {
            lookup.get(raw.as_()).ok_or_else(|| {
                Exception::ValueError(format!(
                    "unknown passband {raw}; configured bands are: {}",
                    lookup.sorted_values_display()
                ))
            })
        })
        .collect()
}

/// Try to parse an integer band array using typed `PyArray1<T>` slices (zero-copy, no byte dispatch).
/// Returns `None` if the input is not a 1-D C-contiguous numpy integer array.
fn try_int_band_view_lookup(
    band_py: &Bound<PyAny>,
    lookup: &IntBandLookup,
) -> Res<Option<Vec<usize>>> {
    let arr = match band_py.cast::<PyUntypedArray>() {
        Ok(a) if a.ndim() == 1 && a.is_c_contiguous() => a,
        _ => return Ok(None),
    };
    let dtype = arr.dtype();
    // Big-endian arrays give wrong values when read as native-endian typed slices;
    // fall back to Python iteration which extracts each element correctly.
    if dtype.is_native_byteorder() == Some(false) {
        return Ok(None);
    }
    let dtype_char = dtype.char() as char;
    match dtype_char {
        'b' => int_band_view_lookup_typed::<i8>(band_py, lookup).map(Some),
        'h' => int_band_view_lookup_typed::<i16>(band_py, lookup).map(Some),
        'i' => int_band_view_lookup_typed::<i32>(band_py, lookup).map(Some),
        'l' => match dtype.itemsize() {
            4 => int_band_view_lookup_typed::<i32>(band_py, lookup).map(Some),
            8 => int_band_view_lookup_typed::<i64>(band_py, lookup).map(Some),
            _ => Ok(None),
        },
        'q' => int_band_view_lookup_typed::<i64>(band_py, lookup).map(Some),
        'B' => int_band_view_lookup_typed::<u8>(band_py, lookup).map(Some),
        'H' => int_band_view_lookup_typed::<u16>(band_py, lookup).map(Some),
        'I' => int_band_view_lookup_typed::<u32>(band_py, lookup).map(Some),
        'L' => match dtype.itemsize() {
            4 => int_band_view_lookup_typed::<u32>(band_py, lookup).map(Some),
            8 => int_band_view_lookup_typed::<u64>(band_py, lookup).map(Some),
            _ => Ok(None),
        },
        'Q' => int_band_view_lookup_typed::<u64>(band_py, lookup).map(Some),
        _ => Ok(None),
    }
}

/// Slice an Arrow primitive band column and resolve each value to a band index.
fn arrow_int_band_lookup<AT: ArrowPrimitiveType>(
    band_col: &dyn Array,
    lookup: &IntBandLookup,
    start: usize,
    end: usize,
) -> Res<Vec<usize>>
where
    AT::Native: TryFrom<i64> + Copy + PartialOrd + num_traits::CheckedSub + fmt::Display,
    usize: TryFrom<AT::Native>,
{
    let values: &[AT::Native] = band_col.as_primitive::<AT>().values();
    let lookup_n = lookup.native_lookup::<AT::Native>();
    values[start..end]
        .iter()
        .map(|&raw| {
            lookup_n.get(raw).ok_or_else(|| {
                Exception::ValueError(format!(
                    "unknown passband {raw}; configured bands are: {}",
                    lookup.sorted_values_display()
                ))
            })
        })
        .collect()
}

/// Try to parse a band array via one `view("uint8")` Python call (zero-copy buffer sharing).
/// Returns `None` if the input is not a 1-D C-contiguous numpy typed-string array.
fn try_band_view_lookup(band_py: &Bound<PyAny>, lookup: &BandLookup) -> Res<Option<Vec<usize>>> {
    let arr = match band_py.cast::<PyUntypedArray>() {
        Ok(a) if a.ndim() == 1 && a.is_c_contiguous() => a,
        _ => return Ok(None),
    };

    let dtype = arr.dtype();
    let kind = dtype.kind();
    let itemsize = dtype.itemsize();

    match kind {
        b'S' | b'a' | b'U' => {}
        b'O' => return Ok(None), // object array: fall through to Python iteration
        b'i' | b'u' => return Ok(None), // integer array: caller uses integer lookup
        _ => {
            return Err(Exception::TypeError(format!(
                "passbands array has non-string dtype (kind '{}'); expected a string dtype (S or U) or an object array",
                kind as char,
            )));
        }
    }

    // One Python call: view("uint8") reinterprets the existing buffer — no data copy.
    let raw = arr
        .call_method1("view", ("uint8",))?
        .cast_into::<PyArray1<u8>>()
        .map_err(|e| Exception::TypeError(format!("band view cast failed: {e}")))?;
    let data = raw.readonly();
    let bytes = data
        .as_slice()
        .map_err(|_| Exception::ValueError("band array is not contiguous".to_string()))?;

    debug_assert_eq!(bytes.len(), arr.shape()[0] * itemsize);

    // Fast path: whole null-padded items compared as u128 against zero-padded keys.
    let packed_table = match kind {
        b'U' => lookup.ucs4_packed.as_ref(),
        _ => lookup.raw_packed.as_ref(),
    };
    if itemsize <= 16
        && let Some(table) = packed_table
    {
        return bytes
            .chunks_exact(itemsize)
            .map(|chunk| {
                let mut buf = [0u8; 16];
                buf[..itemsize].copy_from_slice(chunk);
                let probe = u128::from_le_bytes(buf);
                table
                    .iter()
                    .find_map(|&(k, i)| (k == probe).then_some(i))
                    .ok_or_else(|| unknown_passband_error(chunk, kind))
            })
            .collect::<Res<Vec<usize>>>()
            .map(Some);
    }

    bytes
        .chunks_exact(itemsize)
        .map(|chunk| {
            if kind == b'U' {
                // UCS-4 LE: 4 bytes per codepoint, null-padded to itemsize.
                // Find the end of actual data (first all-zero codepoint) and look up
                // the raw UCS-4 bytes directly — no decoding needed.
                let end = chunk
                    .chunks_exact(4)
                    .position(|cp| cp.iter().all(|&b| b == 0))
                    .map_or(chunk.len(), |i| i * 4);
                lookup
                    .get_ucs4(&chunk[..end])
                    .ok_or_else(|| unknown_passband_error(chunk, kind))
            } else {
                // S / a: null-padded bytes — strip trailing nulls and look up directly.
                let end = chunk.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
                lookup
                    .get_raw(&chunk[..end])
                    .ok_or_else(|| unknown_passband_error(chunk, kind))
            }
        })
        .collect::<Res<Vec<usize>>>()
        .map(Some)
}

/// Decode a null-padded numpy typed-string item for the "unknown passband" error message.
fn unknown_passband_error(chunk: &[u8], kind: u8) -> Exception {
    let s: String = if kind == b'U' {
        chunk
            .chunks_exact(4)
            .filter_map(|b| char::from_u32(u32::from_le_bytes(b.try_into().unwrap())))
            .take_while(|&c| c != '\0')
            .collect()
    } else {
        let end = chunk.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
        String::from_utf8_lossy(&chunk[..end]).into_owned()
    };
    Exception::ValueError(format!("unknown passband '{s}'"))
}

/// Build a flat [lcf::MultiColorTimeSeries] borrowing the input data arrays, with
/// per-observation passband references derived from `band_idx`.
///
/// Currently unused: every multicolor evaluator in light-curve-feature works on the
/// mapping representation, so [mcts_from_indices] is always the better choice. Kept for
/// the day upstream's `EvaluatorInfo` reports which representation an evaluator needs —
/// call sites should then dispatch between the two constructors based on that flag,
/// because a flat representation rebuilt from the mapping would be band-grouped rather
/// than time-ordered.
#[allow(dead_code)]
fn flat_mcts_from_indices<'a, T>(
    t: ndarray::ArrayView1<'a, T>,
    m: ndarray::ArrayView1<'a, T>,
    sigma: Option<ndarray::ArrayView1<'a, T>>,
    band_idx: &[usize],
    sorted_bands: &'a [Passband],
    w_required: bool,
) -> lcf::MultiColorTimeSeries<'a, Passband, T>
where
    T: Float,
{
    let band_refs: Vec<&'a Passband> = band_idx.iter().map(|&i| &sorted_bands[i]).collect();
    // Skip the 1/σ² computation when the evaluator doesn't use weights. A contiguous
    // ones array keeps the per-band grouping on ndarray's slice fast path.
    let w_ds: lcf::DataSample<T> = match sigma.filter(|_| w_required) {
        Some(sigma) => {
            let mut a = sigma.to_owned();
            a.mapv_inplace(|x| x.powi(-2));
            a.into()
        }
        None => ndarray::Array1::ones(t.len()).into(),
    };
    lcf::MultiColorTimeSeries::from_flat_borrowed(t, m, w_ds, band_refs, sorted_bands)
}

/// Build a mapped [lcf::MultiColorTimeSeries] by gathering observations into per-band
/// buffers in a single pass over the data.
///
/// Every multicolor evaluator works on the mapping representation, so constructing it
/// directly avoids the flat intermediate and the per-observation passband dispatch
/// inside light-curve-feature. When `w_required` is false the weight buffers are not
/// gathered at all ([lcf::TimeSeries::new_without_weight] assumes unity weights).
/// See [flat_mcts_from_indices] for the flat-representation counterpart.
fn mcts_from_indices<'a, T>(
    t: ndarray::ArrayView1<'_, T>,
    m: ndarray::ArrayView1<'_, T>,
    sigma: Option<ndarray::ArrayView1<'_, T>>,
    band_idx: &[usize],
    sorted_bands: &[Passband],
    w_required: bool,
) -> lcf::MultiColorTimeSeries<'a, Passband, T>
where
    T: Float,
{
    let k = sorted_bands.len();
    let cap = band_idx.len().checked_div(k).map_or(0, |c| c + 1);
    match sigma.filter(|_| w_required) {
        Some(sigma) => {
            let mut bufs: Vec<(Vec<T>, Vec<T>, Vec<T>)> = (0..k)
                .map(|_| {
                    (
                        Vec::with_capacity(cap),
                        Vec::with_capacity(cap),
                        Vec::with_capacity(cap),
                    )
                })
                .collect();
            for (((&t_val, &m_val), &sigma_val), &idx) in
                t.iter().zip(m.iter()).zip(sigma.iter()).zip(band_idx)
            {
                let (t_buf, m_buf, w_buf) = &mut bufs[idx];
                t_buf.push(t_val);
                m_buf.push(m_val);
                w_buf.push(sigma_val.powi(-2));
            }
            lcf::MultiColorTimeSeries::from_map(
                sorted_bands
                    .iter()
                    .zip(bufs)
                    .filter(|(_, (t_buf, _, _))| !t_buf.is_empty())
                    .map(|(p, (t_buf, m_buf, w_buf))| {
                        (p.clone(), lcf::TimeSeries::new(t_buf, m_buf, w_buf))
                    })
                    .collect::<BTreeMap<_, _>>(),
            )
        }
        None => {
            let mut bufs: Vec<(Vec<T>, Vec<T>)> = (0..k)
                .map(|_| (Vec::with_capacity(cap), Vec::with_capacity(cap)))
                .collect();
            for ((&t_val, &m_val), &idx) in t.iter().zip(m.iter()).zip(band_idx) {
                let (t_buf, m_buf) = &mut bufs[idx];
                t_buf.push(t_val);
                m_buf.push(m_val);
            }
            lcf::MultiColorTimeSeries::from_map(
                sorted_bands
                    .iter()
                    .zip(bufs)
                    .filter(|(_, (t_buf, _))| !t_buf.is_empty())
                    .map(|(p, (t_buf, m_buf))| {
                        (p.clone(), lcf::TimeSeries::new_without_weight(t_buf, m_buf))
                    })
                    .collect::<BTreeMap<_, _>>(),
            )
        }
    }
}

impl PyFeatureEvaluator {
    fn single_band(
        fe_f32: Feature<f32>,
        fe_f64: Feature<f64>,
        transform: Option<Bound<PyAny>>,
        default_transformer: StockTransformer,
    ) -> Res<Self> {
        let transform = parse_transform(transform, default_transformer)?;
        let (fe_f32, fe_f64) = match transform {
            Some(transform) => {
                let (tr_f32, tr_f64) = transform.into();
                let fe_f32 = lcf::Transformed::new(fe_f32, tr_f32)
                    .map_err(|err| {
                        Exception::ValueError(format!(
                            "feature and transformation are incompatible: {err:?}"
                        ))
                    })?
                    .into();
                let fe_f64 = lcf::Transformed::new(fe_f64, tr_f64)
                    .map_err(|err| {
                        Exception::ValueError(format!(
                            "feature and transformation are incompatible: {err:?}"
                        ))
                    })?
                    .into();
                (fe_f32, fe_f64)
            }
            None => (fe_f32, fe_f64),
        };
        Ok(Self {
            mode: FeatureEvalMode::SingleBand {
                feature_evaluator_f32: fe_f32,
                feature_evaluator_f64: fe_f64,
            },
        })
    }

    fn single_band_with_transform(
        (fe_f32, fe_f64): (Feature<f32>, Feature<f64>),
        (tr_f32, tr_f64): (lcf::Transformer<f32>, lcf::Transformer<f64>),
    ) -> Res<Self> {
        Ok(Self {
            mode: FeatureEvalMode::SingleBand {
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
            },
        })
    }

    fn rebuild_band_lookup(&mut self) {
        match &mut self.mode {
            FeatureEvalMode::MultiBand {
                bands,
                sorted_bands,
                band_input,
                ..
            }
            | FeatureEvalMode::Mixed {
                bands,
                sorted_bands,
                band_input,
                ..
            } => {
                let (new_sorted, new_input) = make_sorted_bands_and_input(bands);
                *sorted_bands = new_sorted;
                *band_input = new_input;
            }
            FeatureEvalMode::SingleBand { .. } => {}
        }
    }

    fn multi_band(
        bands: Vec<Passband>,
        mc_f32: lcf::MultiColorFeature<Passband, f32>,
        mc_f64: lcf::MultiColorFeature<Passband, f64>,
    ) -> Self {
        let (sorted_bands, band_input) = make_sorted_bands_and_input(&bands);
        Self {
            mode: FeatureEvalMode::MultiBand {
                band_input,
                sorted_bands,
                bands,
                feature_evaluator_f32: Box::new(mc_f32),
                feature_evaluator_f64: Box::new(mc_f64),
            },
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
        if let Some(sigma) = sigma
            && t.len() != sigma.len()
        {
            return Err(Exception::ValueError(
                "t and sigma must have the same size".to_string(),
            ));
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

    fn many_multiband_impl<T>(
        evaluator: &lcf::MultiColorFeature<Passband, T>,
        lcs: Vec<OwnedMultibandLc<T>>,
        check: bool,
        fill_value: Option<T>,
        n_jobs: i64,
        uniq_bands: &[Passband],
    ) -> Res<ndarray::Array2<T>>
    where
        T: Float + numpy::Element,
    {
        if check {
            for (t, m, sigma, _) in &lcs {
                check_finite(t.view())?;
                check_finite(m.view())?;
                if let Some(s) = sigma {
                    check_no_nans(s.view())?;
                }
            }
        }
        let n_jobs = if n_jobs < 0 { 0 } else { n_jobs as usize };
        let size = evaluator.size_hint();
        let results: Res<Vec<Vec<T>>> = rayon::ThreadPoolBuilder::new()
            .num_threads(n_jobs)
            .build()
            .unwrap()
            .install(|| {
                lcs.into_par_iter()
                    .map(|(t, m, sigma, band_idx)| {
                        let mut mcts = mcts_from_indices(
                            t.view(),
                            m.view(),
                            sigma.as_ref().map(|s| s.view()),
                            &band_idx,
                            uniq_bands,
                            evaluator.is_w_required(),
                        );
                        match fill_value {
                            Some(fv) => evaluator
                                .eval_or_fill_multicolor(&mut mcts, fv)
                                .map_err(|e| Exception::ValueError(format!("{e:?}"))),
                            None => evaluator
                                .eval_multicolor(&mut mcts)
                                .map_err(|e| Exception::ValueError(format!("{e:?}"))),
                        }
                    })
                    .collect()
            });
        let results = results?;
        let n_lcs = results.len();
        let flat: Vec<T> = results.into_iter().flatten().collect();
        ndarray::Array2::from_shape_vec((n_lcs, size), flat)
            .map_err(|e| Exception::ValueError(e.to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    fn py_many_multiband<'py, T>(
        &self,
        evaluator: &lcf::MultiColorFeature<Passband, T>,
        py: Python<'py>,
        lcs: PyMultibandLcs<'py>,
        sorted: Option<bool>,
        check: bool,
        fill_value: Option<T>,
        n_jobs: i64,
    ) -> Res<Bound<'py, PyUntypedArray>>
    where
        T: Float + numpy::Element,
    {
        let (bands, band_input) = match &self.mode {
            FeatureEvalMode::MultiBand {
                sorted_bands,
                band_input,
                ..
            }
            | FeatureEvalMode::Mixed {
                sorted_bands,
                band_input,
                ..
            } => (sorted_bands.as_slice(), band_input),
            FeatureEvalMode::SingleBand { .. } => {
                unreachable!("py_many_multiband called in single-band mode")
            }
        };
        if sorted == Some(false) {
            return Err(Exception::NotImplementedError(
                "sorted=False is not supported in multiband mode".to_string(),
            ));
        }
        let wrapped = lcs
            .into_iter()
            .enumerate()
            .map(|(i, (t, m, sigma, band))| {
                let t = t.cast::<PyArray1<T>>().map(|a| a.readonly());
                let m = m.cast::<PyArray1<T>>().map(|a| a.readonly());
                let sigma = match &sigma {
                    Some(s) => s.cast::<PyArray1<T>>().map(|a| Some(a.readonly())),
                    None => Ok(None),
                };
                match (t, m, sigma) {
                    (Ok(t), Ok(m), Ok(sigma)) => {
                        let band_idx = band_array_to_indices(&band, bands, band_input)?;
                        Ok((
                            t.as_array().to_owned(),
                            m.as_array().to_owned(),
                            sigma.map(|s| s.as_array().to_owned()),
                            band_idx,
                        ))
                    }
                    _ => Err(Exception::TypeError(format!(
                        "lcs[{i}] elements have mismatched dtype with lcs[0][0]"
                    ))),
                }
            })
            .collect::<Res<Vec<_>>>()?;
        Ok(
            Self::many_multiband_impl(evaluator, wrapped, check, fill_value, n_jobs, bands)?
                .into_pyarray(py)
                .as_untyped()
                .clone(),
        )
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
        let (is_t_req, is_sorting_req) = match &self.mode {
            FeatureEvalMode::SingleBand {
                feature_evaluator_f64,
                ..
            } => (
                feature_evaluator_f64.is_t_required(),
                feature_evaluator_f64.is_sorting_required(),
            ),
            FeatureEvalMode::MultiBand {
                feature_evaluator_f64,
                ..
            } => (
                feature_evaluator_f64.is_t_required(),
                feature_evaluator_f64.is_sorting_required(),
            ),
            FeatureEvalMode::Mixed { sb_f64, .. } => {
                (sb_f64.is_t_required(), sb_f64.is_sorting_required())
            }
        };
        match (is_t_req, is_sorting_req, sorted) {
            (true, _, _) => true,
            (false, true, Some(false)) | (false, true, None) => true,
            (false, true, Some(true)) => false,
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
        arrow_fields: &PyArrowFields,
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
        let schema = validate_arrow_lcs(&chunked, arrow_fields)?;
        let is_t_required = self.is_t_required(sorted);

        match &self.mode {
            FeatureEvalMode::SingleBand {
                feature_evaluator_f32,
                feature_evaluator_f64,
            } => match schema.dtype {
                ArrowDtype::F32 => {
                    let result = Self::many_arrow_impl(
                        feature_evaluator_f32,
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
                        feature_evaluator_f64,
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
            },
            FeatureEvalMode::MultiBand {
                sorted_bands,
                band_input,
                feature_evaluator_f32,
                feature_evaluator_f64,
                ..
            } => match schema.dtype {
                ArrowDtype::F32 => {
                    let result = Self::many_arrow_multiband_impl(
                        feature_evaluator_f32,
                        sorted_bands,
                        band_input,
                        &chunked,
                        &schema,
                        sorted,
                        check,
                        fill_value.map(|v| v as f32),
                        n_jobs,
                    )?;
                    Ok(result.into_pyarray(py).as_untyped().clone())
                }
                ArrowDtype::F64 => {
                    let result = Self::many_arrow_multiband_impl(
                        feature_evaluator_f64,
                        sorted_bands,
                        band_input,
                        &chunked,
                        &schema,
                        sorted,
                        check,
                        fill_value,
                        n_jobs,
                    )?;
                    Ok(result.into_pyarray(py).as_untyped().clone())
                }
            },
            FeatureEvalMode::Mixed { .. } => Err(Exception::NotImplementedError(
                "many() with Arrow input is not yet supported for mixed (single+multi band) features".to_string(),
            )),
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
                schema.t_idx,
                schema.m_idx,
                schema.sigma_idx,
                sorted,
                check,
                is_t_required,
                fill_value,
                n_jobs,
            ),
            ArrowListType::LargeList => Self::many_arrow_chunks::<T, i64>(
                feature_evaluator,
                chunked,
                schema.t_idx,
                schema.m_idx,
                schema.sigma_idx,
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
        t_idx: usize,
        m_idx: usize,
        sigma_idx: Option<usize>,
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
                for &col_idx in &[Some(t_idx), Some(m_idx), sigma_idx]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>()
                {
                    if struct_arr.column(col_idx).null_count() > 0 {
                        return Err(Exception::NotImplementedError(
                            "Null values in data columns are not supported".to_string(),
                        ));
                    }
                }

                let t_vals: &[T] = struct_arr
                    .column(t_idx)
                    .as_primitive::<T::ArrowType>()
                    .values()
                    .as_ref();
                let m_vals: &[T] = struct_arr
                    .column(m_idx)
                    .as_primitive::<T::ArrowType>()
                    .values()
                    .as_ref();
                let sigma_vals: Option<&[T]> = sigma_idx.map(|idx| {
                    struct_arr
                        .column(idx)
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

    #[allow(clippy::too_many_arguments)]
    fn many_arrow_multiband_impl<T: ArrowFloat>(
        evaluator: &lcf::MultiColorFeature<Passband, T>,
        bands: &[Passband],
        band_input: &BandInput,
        chunked: &PyChunkedArray,
        schema: &ArrowLcsSchema,
        sorted: Option<bool>,
        check: bool,
        fill_value: Option<T>,
        n_jobs: i64,
    ) -> Res<ndarray::Array2<T>> {
        if sorted == Some(false) {
            return Err(Exception::NotImplementedError(
                "sorted=False is not supported in multiband mode".to_string(),
            ));
        }
        let band_idx = schema.band_idx.ok_or_else(|| {
            Exception::ValueError(
                "arrow_fields must contain \"band\" for multiband features".to_string(),
            )
        })?;
        let band_type = schema.band_type.clone().ok_or_else(|| {
            Exception::ValueError("band_type must be set when band_idx is set".to_string())
        })?;
        match schema.list_type {
            ArrowListType::List => Self::many_arrow_multiband_chunks::<T, i32>(
                evaluator,
                bands,
                band_input,
                chunked,
                schema.t_idx,
                schema.m_idx,
                schema.sigma_idx,
                band_idx,
                band_type,
                check,
                fill_value,
                n_jobs,
            ),
            ArrowListType::LargeList => Self::many_arrow_multiband_chunks::<T, i64>(
                evaluator,
                bands,
                band_input,
                chunked,
                schema.t_idx,
                schema.m_idx,
                schema.sigma_idx,
                band_idx,
                band_type,
                check,
                fill_value,
                n_jobs,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn many_arrow_multiband_chunks<T: ArrowFloat, O: arrow_array::OffsetSizeTrait>(
        evaluator: &lcf::MultiColorFeature<Passband, T>,
        bands: &[Passband],
        band_input: &BandInput,
        chunked: &PyChunkedArray,
        t_idx: usize,
        m_idx: usize,
        sigma_idx: Option<usize>,
        band_idx: usize,
        band_type: ArrowBandType,
        check: bool,
        fill_value: Option<T>,
        n_jobs: i64,
    ) -> Res<ndarray::Array2<T>> {
        let n_jobs_usize = if n_jobs < 0 { 0 } else { n_jobs as usize };
        let size = evaluator.size_hint();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_jobs_usize)
            .build()
            .unwrap();

        // Process one Arrow chunk at a time so views into each chunk's buffer
        // stay alive for the parallel execution within that chunk (zero-copy).
        let mut all_results: Vec<Vec<T>> = Vec::new();
        for chunk in chunked.chunks() {
            let list: &arrow_array::GenericListArray<O> = chunk.as_list::<O>();
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
            for &col_idx in [Some(t_idx), Some(m_idx), sigma_idx, Some(band_idx)]
                .iter()
                .flatten()
            {
                if struct_arr.column(col_idx).null_count() > 0 {
                    return Err(Exception::NotImplementedError(
                        "Null values in data columns are not supported".to_string(),
                    ));
                }
            }

            let t_vals: &[T] = struct_arr
                .column(t_idx)
                .as_primitive::<T::ArrowType>()
                .values()
                .as_ref();
            let m_vals: &[T] = struct_arr
                .column(m_idx)
                .as_primitive::<T::ArrowType>()
                .values()
                .as_ref();
            let sigma_vals: Option<&[T]> = sigma_idx.map(|idx| {
                struct_arr
                    .column(idx)
                    .as_primitive::<T::ArrowType>()
                    .values()
                    .as_ref()
            });

            // Collect (start, end) ranges — O(n_lcs) allocation of usize pairs, nothing more.
            let offsets = list.value_offsets();
            let ranges: Vec<(usize, usize)> = offsets
                .iter()
                .tuple_windows()
                .map(|(&s, &e)| (s.as_usize(), e.as_usize()))
                .collect();

            // Inside rayon: band lookup, checks, and feature evaluation all in parallel.
            let chunk_results: Res<Vec<Vec<T>>> = pool.install(|| {
                ranges
                    .into_par_iter()
                    .map(|(start, end)| {
                        let t = ndarray::ArrayView1::from(&t_vals[start..end]);
                        let m = ndarray::ArrayView1::from(&m_vals[start..end]);
                        let sigma = sigma_vals.map(|s| ndarray::ArrayView1::from(&s[start..end]));
                        let band_indices: Vec<usize> = match (&band_type, band_input) {
                            (ArrowBandType::LargeUtf8, BandInput::String(lookup)) => {
                                let band_arr = struct_arr.column(band_idx).as_string::<i64>();
                                (start..end)
                                    .map(|i| lookup_str_band(lookup, bands, band_arr.value(i)))
                                    .collect::<Res<_>>()?
                            }
                            (ArrowBandType::Utf8, BandInput::String(lookup)) => {
                                let band_arr = struct_arr.column(band_idx).as_string::<i32>();
                                (start..end)
                                    .map(|i| lookup_str_band(lookup, bands, band_arr.value(i)))
                                    .collect::<Res<_>>()?
                            }
                            (ArrowBandType::Utf8View, BandInput::String(lookup)) => {
                                let band_arr = struct_arr.column(band_idx).as_string_view();
                                (start..end)
                                    .map(|i| lookup_str_band(lookup, bands, band_arr.value(i)))
                                    .collect::<Res<_>>()?
                            }
                            (int_arrow_type, BandInput::Integer(lookup)) => {
                                use arrow_array::types::*;
                                let band_col = struct_arr.column(band_idx);
                                match int_arrow_type {
                                    ArrowBandType::Int8   => arrow_int_band_lookup::<Int8Type>(band_col, lookup, start, end)?,
                                    ArrowBandType::Int16  => arrow_int_band_lookup::<Int16Type>(band_col, lookup, start, end)?,
                                    ArrowBandType::Int32  => arrow_int_band_lookup::<Int32Type>(band_col, lookup, start, end)?,
                                    ArrowBandType::Int64  => arrow_int_band_lookup::<Int64Type>(band_col, lookup, start, end)?,
                                    ArrowBandType::UInt8  => arrow_int_band_lookup::<UInt8Type>(band_col, lookup, start, end)?,
                                    ArrowBandType::UInt16 => arrow_int_band_lookup::<UInt16Type>(band_col, lookup, start, end)?,
                                    ArrowBandType::UInt32 => arrow_int_band_lookup::<UInt32Type>(band_col, lookup, start, end)?,
                                    ArrowBandType::UInt64 => arrow_int_band_lookup::<UInt64Type>(band_col, lookup, start, end)?,
                                    _ => unreachable!("integer band_input but string arrow band type"),
                                }
                            }
                            _ => return Err(Exception::TypeError(
                                "band column type (string/integer) does not match feature's band mode".to_string()
                            )),
                        };
                        if check {
                            check_finite(t)?;
                            check_finite(m)?;
                            if let Some(s) = sigma {
                                check_no_nans(s)?;
                            }
                        }
                        let mut mcts = mcts_from_indices(
                            t,
                            m,
                            sigma,
                            &band_indices,
                            bands,
                            evaluator.is_w_required(),
                        );
                        match fill_value {
                            Some(fv) => evaluator
                                .eval_or_fill_multicolor(&mut mcts, fv)
                                .map_err(|e| Exception::ValueError(format!("{e:?}"))),
                            None => evaluator
                                .eval_multicolor(&mut mcts)
                                .map_err(|e| Exception::ValueError(format!("{e:?}"))),
                        }
                    })
                    .collect()
            });
            all_results.extend(chunk_results?);
        }

        if all_results.is_empty() {
            return Err(Exception::ValueError("lcs is empty".to_string()));
        }

        let n_lcs = all_results.len();
        let flat: Vec<T> = all_results.into_iter().flatten().collect();
        ndarray::Array2::from_shape_vec((n_lcs, size), flat)
            .map_err(|e| Exception::ValueError(e.to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    fn call_multiband_impl<'py, T>(
        evaluator: &lcf::MultiColorFeature<Passband, T>,
        py: Python<'py>,
        t: Arr<'py, T>,
        m: Arr<'py, T>,
        sigma: Option<Arr<'py, T>>,
        band_idx: &[usize],
        sorted_bands: &[Passband],
        sorted: Option<bool>,
        check: bool,
        fill_value: Option<T>,
    ) -> Res<Bound<'py, PyUntypedArray>>
    where
        T: Float + numpy::Element,
    {
        let result = Self::call_multiband_impl_vec(
            evaluator,
            t,
            m,
            sigma,
            band_idx,
            sorted_bands,
            sorted,
            check,
            fill_value,
        )?;
        Ok(PyArray1::from_vec(py, result).as_untyped().clone())
    }

    // Lower-level helpers that return raw `Vec<T>` so that mixed mode can interleave results.
    #[allow(clippy::too_many_arguments)]
    fn call_impl_vec<T>(
        feature_evaluator: &Feature<T>,
        t: Arr<T>,
        m: Arr<T>,
        sigma: Option<Arr<T>>,
        sorted: Option<bool>,
        check: bool,
        is_t_required: bool,
        fill_value: Option<T>,
    ) -> Res<Vec<T>>
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
        Ok(match fill_value {
            Some(x) => feature_evaluator.eval_or_fill(&mut ts, x),
            None => feature_evaluator
                .eval(&mut ts)
                .map_err(|e| Exception::ValueError(e.to_string()))?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn call_multiband_impl_vec<T>(
        evaluator: &lcf::MultiColorFeature<Passband, T>,
        t: Arr<T>,
        m: Arr<T>,
        sigma: Option<Arr<T>>,
        band_idx: &[usize],
        sorted_bands: &[Passband],
        sorted: Option<bool>,
        check: bool,
        fill_value: Option<T>,
    ) -> Res<Vec<T>>
    where
        T: Float + numpy::Element,
    {
        if sorted == Some(false) {
            return Err(Exception::NotImplementedError(
                "sorted=False is not supported in multiband mode".to_string(),
            ));
        }
        let n = t.len();
        if n != m.len() {
            return Err(Exception::ValueError(
                "t and m must have the same size".to_string(),
            ));
        }
        if let Some(s) = &sigma
            && n != s.len()
        {
            return Err(Exception::ValueError(
                "t and sigma must have the same size".to_string(),
            ));
        }
        if n != band_idx.len() {
            return Err(Exception::ValueError(
                "t and band must have the same size".to_string(),
            ));
        }
        if check {
            check_finite(t.as_array())?;
            check_finite(m.as_array())?;
            if let Some(s) = &sigma {
                check_no_nans(s.as_array())?;
            }
        }
        let mut mcts = mcts_from_indices(
            t.as_array(),
            m.as_array(),
            sigma.as_ref().map(|s| s.as_array()),
            band_idx,
            sorted_bands,
            evaluator.is_w_required(),
        );
        Ok(match fill_value {
            Some(fv) => evaluator
                .eval_or_fill_multicolor(&mut mcts, fv)
                .map_err(|e| Exception::ValueError(format!("{e:?}")))?,
            None => evaluator
                .eval_multicolor(&mut mcts)
                .map_err(|e| Exception::ValueError(format!("{e:?}")))?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn call_mixed_impl<'py, T>(
        sb_eval: &Feature<T>,
        mc_eval: &lcf::MultiColorFeature<Passband, T>,
        py: Python<'py>,
        t: Arr<'py, T>,
        m: Arr<'py, T>,
        sigma: Option<Arr<'py, T>>,
        band_idx: &[usize],
        sorted_bands: &[Passband],
        sb_mask: &[bool],
        sorted: Option<bool>,
        check: bool,
        sb_is_t_required: bool,
        fill_value: Option<T>,
    ) -> Res<Bound<'py, PyUntypedArray>>
    where
        T: Float + numpy::Element,
    {
        // Clone readonly refs so we can pass each to a separate helper.
        // Safety: PyReadonlyArray is a shared borrow of a Python object; cloning
        // the reference is fine because both calls are read-only.
        let t2 = t.as_array().to_owned();
        let m2 = m.as_array().to_owned();
        let sigma2: Option<ndarray::Array1<T>> = sigma.as_ref().map(|s| s.as_array().to_owned());

        let sb_results = Self::call_impl_vec(
            sb_eval,
            t,
            m,
            sigma,
            sorted,
            check,
            sb_is_t_required,
            fill_value,
        )?;

        // Build Arr-like owned arrays for the mc path.
        let t_arr = PyArray1::from_array(py, &t2.view());
        let m_arr = PyArray1::from_array(py, &m2.view());
        let sigma_arr = sigma2.as_ref().map(|s| PyArray1::from_array(py, &s.view()));

        let mc_results = Self::call_multiband_impl_vec(
            mc_eval,
            t_arr.readonly(),
            m_arr.readonly(),
            sigma_arr.as_ref().map(|a| a.readonly()),
            band_idx,
            sorted_bands,
            sorted,
            check,
            fill_value,
        )?;

        // Interleave sb and mc results according to sb_mask.
        let mut result = Vec::with_capacity(sb_mask.len());
        let mut sb_iter = sb_results.into_iter();
        let mut mc_iter = mc_results.into_iter();
        for &is_sb in sb_mask {
            result.push(if is_sb {
                sb_iter.next().unwrap()
            } else {
                mc_iter.next().unwrap()
            });
        }
        Ok(PyArray1::from_vec(py, result).as_untyped().clone())
    }
}

#[pymethods]
impl PyFeatureEvaluator {
    #[doc = METHOD_CALL_DOC!()]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        t,
        m,
        sigma = None,
        band = None,
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
        band: Option<Bound<'py, PyAny>>,
        fill_value: Option<f64>,
        sorted: Option<bool>,
        check: bool,
        cast: bool,
    ) -> Res<Bound<'py, PyUntypedArray>> {
        match &self.mode {
            FeatureEvalMode::MultiBand {
                sorted_bands,
                feature_evaluator_f32: mc_f32,
                feature_evaluator_f64: mc_f64,
                band_input,
                ..
            } => {
                let band_py = band.ok_or_else(|| {
                    Exception::ValueError(
                        "band must be provided when bands is not None (multiband mode)".to_string(),
                    )
                })?;
                let band_idx = band_array_to_indices(&band_py, sorted_bands, band_input)?;
                if let Some(sigma) = sigma {
                    dtype_dispatch!(
                        |t, m, sigma| Self::call_multiband_impl(mc_f32, py, t, m, Some(sigma), &band_idx, sorted_bands, sorted, check, fill_value.map(|v| v as f32)),
                        |t, m, sigma| Self::call_multiband_impl(mc_f64, py, t, m, Some(sigma), &band_idx, sorted_bands, sorted, check, fill_value),
                        t, =m, =sigma; cast=cast
                    )
                } else {
                    dtype_dispatch!(
                        |t, m| Self::call_multiband_impl(mc_f32, py, t, m, None, &band_idx, sorted_bands, sorted, check, fill_value.map(|v| v as f32)),
                        |t, m| Self::call_multiband_impl(mc_f64, py, t, m, None, &band_idx, sorted_bands, sorted, check, fill_value),
                        t, =m; cast=cast
                    )
                }
            }
            FeatureEvalMode::SingleBand {
                feature_evaluator_f32,
                feature_evaluator_f64,
            } => {
                if let Some(sigma) = sigma {
                    dtype_dispatch!(
                        |t, m, sigma| {
                            Self::call_impl(
                                feature_evaluator_f32,
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
                                feature_evaluator_f64,
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
                                feature_evaluator_f32,
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
                                feature_evaluator_f64,
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
            FeatureEvalMode::Mixed {
                sorted_bands,
                sb_f32,
                sb_f64,
                mc_f32,
                mc_f64,
                sb_mask,
                band_input,
                ..
            } => {
                let band_py = band.ok_or_else(|| {
                    Exception::ValueError(
                        "band must be provided when bands is not None (multiband mode)".to_string(),
                    )
                })?;
                let band_idx = band_array_to_indices(&band_py, sorted_bands, band_input)?;
                let sb_is_t_required = self.is_t_required(sorted);
                if let Some(sigma) = sigma {
                    dtype_dispatch!(
                        |t, m, sigma| Self::call_mixed_impl(sb_f32, mc_f32, py, t, m, Some(sigma), &band_idx, sorted_bands, sb_mask, sorted, check, sb_is_t_required, fill_value.map(|v| v as f32)),
                        |t, m, sigma| Self::call_mixed_impl(sb_f64, mc_f64, py, t, m, Some(sigma), &band_idx, sorted_bands, sb_mask, sorted, check, sb_is_t_required, fill_value),
                        t, =m, =sigma; cast=cast
                    )
                } else {
                    dtype_dispatch!(
                        |t, m| Self::call_mixed_impl(sb_f32, mc_f32, py, t, m, None, &band_idx, sorted_bands, sb_mask, sorted, check, sb_is_t_required, fill_value.map(|v| v as f32)),
                        |t, m| Self::call_mixed_impl(sb_f64, mc_f64, py, t, m, None, &band_idx, sorted_bands, sb_mask, sorted, check, sb_is_t_required, fill_value),
                        t, =m; cast=cast
                    )
                }
            }
        }
    }

    #[doc = METHOD_MANY_DOC!()]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (lcs, *, fill_value=None, sorted=None, check=true, n_jobs=-1, arrow_fields=None))]
    fn many<'py>(
        &self,
        py: Python<'py>,
        lcs: Bound<'py, PyAny>,
        fill_value: Option<f64>,
        sorted: Option<bool>,
        check: bool,
        n_jobs: i64,
        arrow_fields: Option<PyArrowFields>,
    ) -> Res<Bound<'py, PyUntypedArray>> {
        // Arrow path: auto-detected, handles all modes
        if lcs.hasattr("__arrow_c_array__")? || lcs.hasattr("__arrow_c_stream__")? {
            let fields = arrow_fields.ok_or_else(|| {
                Exception::ValueError(
                    "arrow_fields is required when using Arrow input; \
                    provide a dict e.g. {\"t\": \"time\", \"m\": \"flux\", \"sigma\": \"fluxerr\", \"band\": \"passband\"} \
                    (\"sigma\" and \"band\" are optional)"
                        .to_string(),
                )
            })?;
            return self.many_arrow(py, &lcs, fill_value, sorted, check, n_jobs, &fields);
        }
        // List-of-tuples path
        if let FeatureEvalMode::MultiBand {
            feature_evaluator_f32: mc_f32,
            feature_evaluator_f64: mc_f64,
            ..
        } = &self.mode
        {
            let lcs: PyMultibandLcs<'py> = lcs.extract()?;
            if lcs.is_empty() {
                return Err(Exception::ValueError("lcs is empty".to_string()));
            }
            return dtype_dispatch!(
                |_first_t| {
                    self.py_many_multiband(
                        mc_f32,
                        py,
                        lcs,
                        sorted,
                        check,
                        fill_value.map(|v| v as f32),
                        n_jobs,
                    )
                },
                |_first_t| {
                    self.py_many_multiband(mc_f64, py, lcs, sorted, check, fill_value, n_jobs)
                },
                lcs[0].0
            );
        }
        if matches!(&self.mode, FeatureEvalMode::Mixed { .. }) {
            return Err(Exception::NotImplementedError(
                "many() is not yet supported for mixed-mode features".to_string(),
            ));
        }
        let (feature_evaluator_f32, feature_evaluator_f64) = match &self.mode {
            FeatureEvalMode::SingleBand {
                feature_evaluator_f32,
                feature_evaluator_f64,
            } => (feature_evaluator_f32, feature_evaluator_f64),
            FeatureEvalMode::MultiBand { .. } | FeatureEvalMode::Mixed { .. } => unreachable!(),
        };
        // Fall back to list-of-tuples path
        let lcs: PyLcs<'py> = lcs.extract()?;
        if lcs.is_empty() {
            Err(Exception::ValueError("lcs is empty".to_string()))
        } else {
            dtype_dispatch!(
                |_first_t| {
                    self.py_many(
                        feature_evaluator_f32,
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
                        feature_evaluator_f64,
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
    fn to_json(&self) -> Res<String> {
        #[derive(Serialize)]
        struct MultiBandJson<'a> {
            bands: &'a Vec<Passband>,
            feature: &'a lcf::MultiColorFeature<Passband, f64>,
        }

        match &self.mode {
            FeatureEvalMode::SingleBand {
                feature_evaluator_f64,
                ..
            } => Ok(serde_json::to_string(feature_evaluator_f64).unwrap()),
            FeatureEvalMode::MultiBand {
                bands,
                feature_evaluator_f64,
                ..
            } => Ok(serde_json::to_string(&MultiBandJson {
                bands,
                feature: feature_evaluator_f64,
            })
            .unwrap()),
            FeatureEvalMode::Mixed { .. } => Err(Exception::NotImplementedError(
                "to_json() is not supported for mixed-mode features".to_string(),
            )),
        }
    }

    /// Feature names
    #[getter]
    fn names(&self) -> Vec<&str> {
        match &self.mode {
            FeatureEvalMode::SingleBand {
                feature_evaluator_f64,
                ..
            } => feature_evaluator_f64.get_names(),
            FeatureEvalMode::MultiBand {
                feature_evaluator_f64,
                ..
            } => feature_evaluator_f64.get_names(),
            FeatureEvalMode::Mixed {
                sb_f64,
                mc_f64,
                sb_mask,
                ..
            } => {
                let mut sb = sb_f64.get_names().into_iter();
                let mut mc = mc_f64.get_names().into_iter();
                sb_mask
                    .iter()
                    .map(|&is_sb| if is_sb { sb.next() } else { mc.next() }.unwrap())
                    .collect()
            }
        }
    }

    /// Feature descriptions
    #[getter]
    fn descriptions(&self) -> Vec<&str> {
        match &self.mode {
            FeatureEvalMode::SingleBand {
                feature_evaluator_f64,
                ..
            } => feature_evaluator_f64.get_descriptions(),
            FeatureEvalMode::MultiBand {
                feature_evaluator_f64,
                ..
            } => feature_evaluator_f64.get_descriptions(),
            FeatureEvalMode::Mixed {
                sb_f64,
                mc_f64,
                sb_mask,
                ..
            } => {
                let mut sb = sb_f64.get_descriptions().into_iter();
                let mut mc = mc_f64.get_descriptions().into_iter();
                sb_mask
                    .iter()
                    .map(|&is_sb| if is_sb { sb.next() } else { mc.next() }.unwrap())
                    .collect()
            }
        }
    }

    /// Passband names/IDs (multiband mode only).
    /// Returns a numpy array of str for string bands, or numpy array of int64 for integer bands.
    #[getter]
    fn bands<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let np = PyModule::import(py, "numpy")?;
        match &self.mode {
            FeatureEvalMode::SingleBand { .. } => Ok(None),
            FeatureEvalMode::MultiBand { bands, .. } | FeatureEvalMode::Mixed { bands, .. } => {
                match bands.first() {
                    Some(Passband::Integer(_)) => {
                        let ints: Vec<i64> = bands
                            .iter()
                            .map(|p| match p {
                                Passband::Integer(lp) => lp.label,
                                Passband::String(_) => unreachable!(),
                            })
                            .collect();
                        Ok(Some(np.getattr("array")?.call1((ints,))?))
                    }
                    _ => {
                        let names: Vec<&str> = bands.iter().map(|p| p.name()).collect();
                        Ok(Some(np.getattr("array")?.call1((names,))?))
                    }
                }
            }
        }
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
                let (mut super_rust, self_rust): (PyFeatureEvaluator, Self) = serde_pickle::from_slice(state.as_bytes(), serde_pickle::DeOptions::new())
                    .map_err(|err| {
                        Exception::UnpicklingError(format!(
                            r#"Error happened on the Rust side when deserializing _FeatureEvaluator: "{err}""#
                        ))
                    })?;
                super_rust.rebuild_band_lookup();
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
        // Classify each feature as single- or multi-band, collecting both flavours.
        enum FeatureItem {
            Single(Feature<f32>, Feature<f64>),
            Multi(
                lcf::MultiColorFeature<Passband, f32>,
                lcf::MultiColorFeature<Passband, f64>,
            ),
        }
        let items: Vec<FeatureItem> = features
            .iter_borrowed()
            .map(|arg| -> PyResult<FeatureItem> {
                let fe = arg.cast::<PyFeatureEvaluator>()?.borrow();
                match &fe.mode {
                    FeatureEvalMode::SingleBand {
                        feature_evaluator_f32,
                        feature_evaluator_f64,
                    } => Ok(FeatureItem::Single(
                        feature_evaluator_f32.clone(),
                        feature_evaluator_f64.clone(),
                    )),
                    FeatureEvalMode::MultiBand {
                        feature_evaluator_f32,
                        feature_evaluator_f64,
                        ..
                    } => Ok(FeatureItem::Multi(
                        *feature_evaluator_f32.clone(),
                        *feature_evaluator_f64.clone(),
                    )),
                    FeatureEvalMode::Mixed { .. } => Err(Exception::ValueError(
                        "Extractor does not support nesting mixed-mode features".to_string(),
                    )
                    .into()),
                }
            })
            .collect::<PyResult<_>>()?;

        let has_single = items.iter().any(|i| matches!(i, FeatureItem::Single(..)));
        let has_multi = items.iter().any(|i| matches!(i, FeatureItem::Multi(..)));

        let parent = if has_single && has_multi {
            let mut sb_f32_evals = Vec::new();
            let mut sb_f64_evals = Vec::new();
            let mut mc_f32_evals = Vec::new();
            let mut mc_f64_evals = Vec::new();
            let mut sb_mask = Vec::new();
            for item in items {
                match item {
                    FeatureItem::Single(f32_eval, f64_eval) => {
                        let size = f64_eval.size_hint();
                        sb_f32_evals.push(f32_eval);
                        sb_f64_evals.push(f64_eval);
                        sb_mask.extend(std::iter::repeat_n(true, size));
                    }
                    FeatureItem::Multi(f32_eval, f64_eval) => {
                        let size = f64_eval.size_hint();
                        mc_f32_evals.push(f32_eval);
                        mc_f64_evals.push(f64_eval);
                        sb_mask.extend(std::iter::repeat_n(false, size));
                    }
                }
            }
            let mc_extractor_f32 = lcf::MultiColorExtractor::new(mc_f32_evals);
            let mc_extractor_f64 = lcf::MultiColorExtractor::new(mc_f64_evals);
            let bands_vec: Vec<Passband> = {
                use lcf::MultiColorPassbandSetTrait;
                mc_extractor_f64
                    .get_passband_set()
                    .0
                    .iter()
                    .cloned()
                    .collect()
            };
            {
                let (sorted_bands, band_input) = make_sorted_bands_and_input(&bands_vec);
                PyFeatureEvaluator {
                    mode: FeatureEvalMode::Mixed {
                        band_input,
                        sorted_bands,
                        bands: bands_vec,
                        sb_f32: FeatureExtractor::new(sb_f32_evals).into(),
                        sb_f64: FeatureExtractor::new(sb_f64_evals).into(),
                        mc_f32: Box::new(lcf::MultiColorFeature::MultiColorExtractor(
                            mc_extractor_f32,
                        )),
                        mc_f64: Box::new(lcf::MultiColorFeature::MultiColorExtractor(
                            mc_extractor_f64,
                        )),
                        sb_mask,
                    },
                }
            }
        } else if has_multi {
            let (mc_f32, mc_f64): (Vec<_>, Vec<_>) = items
                .into_iter()
                .map(|i| match i {
                    FeatureItem::Multi(f32, f64) => (f32, f64),
                    FeatureItem::Single(..) => unreachable!(),
                })
                .unzip();
            let mc_extractor_f32 = lcf::MultiColorExtractor::new(mc_f32);
            let mc_extractor_f64 = lcf::MultiColorExtractor::new(mc_f64);
            let bands_vec: Vec<Passband> = {
                use lcf::MultiColorPassbandSetTrait;
                mc_extractor_f64
                    .get_passband_set()
                    .0
                    .iter()
                    .cloned()
                    .collect()
            };
            PyFeatureEvaluator::multi_band(
                bands_vec,
                lcf::MultiColorFeature::MultiColorExtractor(mc_extractor_f32),
                lcf::MultiColorFeature::MultiColorExtractor(mc_extractor_f64),
            )
        } else {
            let (evals_f32, evals_f64): (Vec<_>, Vec<_>) = items
                .into_iter()
                .map(|i| match i {
                    FeatureItem::Single(f32, f64) => (f32, f64),
                    FeatureItem::Multi(..) => unreachable!(),
                })
                .unzip();
            PyFeatureEvaluator {
                mode: FeatureEvalMode::SingleBand {
                    feature_evaluator_f32: FeatureExtractor::new(evals_f32).into(),
                    feature_evaluator_f64: FeatureExtractor::new(evals_f64).into(),
                },
            }
        };
        Ok((Self {}, parent))
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
            prepare_upstream_doc(FeatureExtractor::<f64, Feature<f64>>::doc()),
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
            #[pyo3(signature=(*, transform=None, bands=None))]
            fn __new__(
                transform: Option<Bound<PyAny>>,
                bands: Option<Bound<'_, PyAny>>,
            ) -> Res<PyClassInitializer<Self>> {
                let base = match bands {
                    None => PyFeatureEvaluator::single_band(
                        <$eval>::new().into(),
                        <$eval>::new().into(),
                        transform,
                        Self::DEFAULT_TRANSFORMER,
                    )?,
                    Some(bands_py) => {
                        let user_bands = parse_bands(&bands_py)?;
                        let mc_f32 = lcf::MultiColorFeature::from_per_band_feature(
                            <$eval>::new(),
                            user_bands.clone(),
                        );
                        let mc_f64 = lcf::MultiColorFeature::from_per_band_feature(
                            <$eval>::new(),
                            user_bands.clone(),
                        );
                        PyFeatureEvaluator::multi_band(user_bands, mc_f32, mc_f64)
                    }
                };
                Ok(PyClassInitializer::from(base).add_subclass(Self {}))
            }

            #[classattr]
            fn __doc__() -> String {
                format!(
                    r#"{header}
Parameters
----------
{transform_variant}
{bands}
{footer}"#,
                    header = prepare_upstream_doc(<$eval>::doc()),
                    transform_variant = transform_parameter_doc(Self::DEFAULT_TRANSFORMER),
                    bands = BANDS_PARAMETER_DOC,
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
const N_ALGO_CURVE_FIT_NUTS: usize = 1 + N_ALGO_CURVE_FIT_CERES / 2 + N_ALGO_CURVE_FIT_GSL / 2;
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
    "nuts",
    #[cfg(any(feature = "ceres-source", feature = "ceres-system"))]
    "nuts-ceres",
    #[cfg(feature = "gsl")]
    "nuts-lmsder",
];

macro_const! {
    const FIT_METHOD_MODEL_DOC: &str = r#"model(t, params, *, cast=False)
    Underlying parametric model function

    Parameters
    ----------

    `t` : np.ndarray of float32 or float64
    :   Time moments, can be unsorted

    `params` : np.ndarray of float32 or float64
    :   Parameters of the model, this array can be longer than actual parameter
        list, the beginning part of the array will be used in this case, see
        Examples section in the class documentation.

    `cast` : bool, optional
    :   Cast inputs to np.ndarray of the same dtype

    Returns
    -------

    np.ndarray of float32 or float64
    :   Array of model values corresponded to the given time moments
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

            fn default_nuts_ntune() -> u32 {
                lcf::NutsCurveFit::default_num_tune()
            }

            fn default_nuts_niter() -> u32 {
                lcf::NutsCurveFit::default_num_draws()
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
                nuts_niter = Self::default_nuts_niter(),
                init = None,
                bounds = None,
                ln_prior = None,
                transform = None,
                bands = None,
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
                nuts_ntune: u32,
                nuts_niter: u32,
                init: Option<Vec<Option<f64>>>,
                bounds: Option<Vec<(Option<f64>, Option<f64>)>>,
                ln_prior: Option<FitLnPrior>,
                transform: Option<Bound<PyAny>>,
                bands: Option<Bound<'_, PyAny>>,
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
                    "nuts" => lcf::NutsCurveFit::new(nuts_ntune, nuts_niter, None).into(),
                    #[cfg(any(feature = "ceres-source", feature = "ceres-system"))]
                    "nuts-ceres" => lcf::NutsCurveFit::new(nuts_ntune, nuts_niter, Some(ceres_fit)).into(),
                    #[cfg(feature = "gsl")]
                    "nuts-lmsder" => lcf::NutsCurveFit::new(nuts_ntune, nuts_niter, Some(lmsder_fit)).into(),
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
                let fe = match bands {
                    None => {
                        if make_transformation {
                            PyFeatureEvaluator::single_band_with_transform((fe_f32, fe_f64), ($transform.into(), $transform.into()))?
                        } else {
                            PyFeatureEvaluator {
                                mode: FeatureEvalMode::SingleBand {
                                    feature_evaluator_f32: fe_f32,
                                    feature_evaluator_f64: fe_f64,
                                },
                            }
                        }
                    }
                    Some(bands_py) => {
                        if make_transformation {
                            return Err(PyValueError::new_err("transform is not supported in multiband mode"));
                        }
                        let user_bands = parse_bands(&bands_py)?;
                        let mc_f32 = lcf::MultiColorFeature::from_per_band_feature(fe_f32, user_bands.clone());
                        let mc_f64 = lcf::MultiColorFeature::from_per_band_feature(fe_f64, user_bands.clone());
                        PyFeatureEvaluator::multi_band(user_bands, mc_f32, mc_f64)
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
                    r#"ceres_niter : int, default {niter}
    Number of Ceres iterations

ceres_loss_reg : float or None, default None
    Ceres loss regularization. If set to a number, the loss function is
    regularized to discriminate outlier residuals larger than this value.
    ``None`` means no regularization.

"#,
                    niter = lcf::CeresCurveFit::default_niterations()
                );
                #[cfg(not(any(feature = "ceres-source", feature = "ceres-system")))]
                let ceres_args = "";
                #[cfg(feature = "gsl")]
                let lmsder_niter = format!(
                    r#"lmsder_niter : int, default {}
    Number of LMSDER iterations

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

mcmc_niter : int, default {mcmc_niter}
    Number of MCMC iterations

{ceres_args}{lmsder_niter}
init : list or None, default None
    Initial conditions, must be ``None`` or a ``list`` of ``float``s or ``None``s.
    The length of the list must be {nparam}; ``None`` values will be replaced
    with some default values. Supported by MCMC only.

bounds : list of tuples or None, default None
    Boundary conditions, must be ``None`` or a ``list`` of ``tuple``s of ``float``s or
    ``None``s. The length of the list must be {nparam}; boundary conditions must
    include initial conditions; ``None`` values will be replaced with some broad
    defaults. Supported by MCMC only.

ln_prior : str or list of ln_prior.LnPrior1D or None, default None
    Prior for MCMC, ``None`` means no prior. Specified by a string literal
    or a list of {nparam} ``ln_prior.LnPrior1D`` objects; see the ``ln_prior``
    submodule for corresponding functions. Available string literals are:
    {ln_prior}

transform : bool or None, default None
    If ``False`` or ``None`` output is not transformed. If ``True`` output
    is transformed as follows:

     - Half-amplitude A is transformed as ``zp - 2.5 lg(2*A)``, zp = 8.9,
       so that the amplitude is assumed to be the object peak flux in Jy.
     - Baseline flux is normalised by A: baseline → baseline / A
     - Reference time is removed
     - Goodness of fit is transformed as ``ln(reduced chi^2 + 1)`` to reduce
       its spread
     - Other parameters are not transformed

    See ``names`` and ``descriptions`` attributes for the list and order
    of features.
{bands_doc}

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
                    intro = prepare_upstream_doc(<$eval>::doc()),
                    names_descriptions = names_descriptions,
                    supported_algo = Self::supported_algorithms_str(),
                    mcmc_niter = lcf::McmcCurveFit::default_niterations(),
                    ceres_args = ceres_args,
                    lmsder_niter = lmsder_niter,
                    bands_doc = BANDS_PARAMETER_DOC,
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
    #[pyo3(signature = (nstd=lcf::BeyondNStd::<f32>::default_nstd(), *, transform=None, bands=None))]
    fn __new__(
        nstd: f32,
        transform: Option<Bound<PyAny>>,
        bands: Option<Bound<'_, PyAny>>,
    ) -> Res<PyClassInitializer<Self>> {
        let base = match bands {
            None => PyFeatureEvaluator::single_band(
                lcf::BeyondNStd::new(nstd).into(),
                lcf::BeyondNStd::new(nstd).into(),
                transform,
                Self::DEFAULT_TRANSFORMER,
            )?,
            Some(bands_py) => {
                let user_bands = parse_bands(&bands_py)?;
                let mc_f32 = lcf::MultiColorFeature::from_per_band_feature(
                    lcf::BeyondNStd::new(nstd),
                    user_bands.clone(),
                );
                let mc_f64 = lcf::MultiColorFeature::from_per_band_feature(
                    lcf::BeyondNStd::new(nstd),
                    user_bands.clone(),
                );
                PyFeatureEvaluator::multi_band(user_bands, mc_f32, mc_f64)
            }
        };
        Ok(PyClassInitializer::from(base).add_subclass(Self {}))
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
nstd : positive float, default {nstd_default:.1}
    N — how many standard deviations from the mean

{transform}
{bands}
{footer}"#,
            header = prepare_upstream_doc(lcf::BeyondNStd::<f64>::doc()),
            nstd_default = lcf::BeyondNStd::<f64>::default_nstd(),
            transform = transform_parameter_doc(Self::DEFAULT_TRANSFORMER),
            bands = BANDS_PARAMETER_DOC,
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
    #[pyo3(signature = (features, *, window, offset, transform = None, bands = None))]
    fn __new__(
        features: Bound<PyAny>,
        window: f64,
        offset: f64,
        transform: Option<Bound<PyAny>>,
        bands: Option<Bound<'_, PyAny>>,
    ) -> PyResult<(Self, PyFeatureEvaluator)> {
        if transform.is_some() {
            return Err(Exception::NotImplementedError(
                "transform is not supported by Bins, apply transformations to individual features"
                    .to_string(),
            )
            .into());
        }

        let parent = match bands {
            None => {
                let mut eval_f32 = lcf::Bins::default();
                let mut eval_f64 = lcf::Bins::default();
                for x in features.try_iter()? {
                    let py_feature = x?.cast::<PyFeatureEvaluator>()?.borrow();
                    let (f32_eval, f64_eval) = match &py_feature.mode {
                        FeatureEvalMode::SingleBand {
                            feature_evaluator_f32,
                            feature_evaluator_f64,
                        } => (feature_evaluator_f32.clone(), feature_evaluator_f64.clone()),
                        FeatureEvalMode::MultiBand { .. } | FeatureEvalMode::Mixed { .. } => {
                            return Err(Exception::ValueError(
                                "Bins without bands= does not support multiband features; pass bands= to enable multiband binning".to_string(),
                            )
                            .into())
                        }
                    };
                    eval_f32.add_feature(f32_eval);
                    eval_f64.add_feature(f64_eval);
                }
                eval_f32.set_window(window);
                eval_f64.set_window(window);
                eval_f32.set_offset(offset);
                eval_f64.set_offset(offset);
                PyFeatureEvaluator {
                    mode: FeatureEvalMode::SingleBand {
                        feature_evaluator_f32: eval_f32.into(),
                        feature_evaluator_f64: eval_f64.into(),
                    },
                }
            }
            Some(bands_py) => {
                let user_bands = parse_bands(&bands_py)?;
                let mut mc_bins_f32 = lcf::MultiColorBins::new(window, offset);
                let mut mc_bins_f64 = lcf::MultiColorBins::new(window, offset);
                for x in features.try_iter()? {
                    let py_feature = x?.cast::<PyFeatureEvaluator>()?.borrow();
                    match &py_feature.mode {
                        FeatureEvalMode::SingleBand {
                            feature_evaluator_f32,
                            feature_evaluator_f64,
                        } => {
                            mc_bins_f32.add_feature(lcf::MultiColorFeature::from_per_band_feature(
                                feature_evaluator_f32.clone(),
                                user_bands.clone(),
                            ));
                            mc_bins_f64.add_feature(lcf::MultiColorFeature::from_per_band_feature(
                                feature_evaluator_f64.clone(),
                                user_bands.clone(),
                            ));
                        }
                        FeatureEvalMode::MultiBand {
                            feature_evaluator_f32,
                            feature_evaluator_f64,
                            ..
                        } => {
                            mc_bins_f32.add_feature(*feature_evaluator_f32.clone());
                            mc_bins_f64.add_feature(*feature_evaluator_f64.clone());
                        }
                        FeatureEvalMode::Mixed { .. } => {
                            return Err(Exception::ValueError(
                                "Bins does not support mixed-mode features".to_string(),
                            )
                            .into());
                        }
                    }
                }
                PyFeatureEvaluator::multi_band(
                    user_bands,
                    lcf::MultiColorFeature::MultiColorBins(mc_bins_f32),
                    lcf::MultiColorFeature::MultiColorBins(mc_bins_f64),
                )
            }
        };

        Ok((Self {}, parent))
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

transform : None, default None
    Not supported, apply transformations to individual features
bands : list of str or None, optional
    Passband names for multiband mode. If given, each single-band feature in
    ``features`` is evaluated independently per passband; multiband features
    (e.g. color features) are passed through unchanged.
{footer}
"#,
            header = prepare_upstream_doc(lcf::Bins::<f64, Feature<f64>>::doc()),
            footer = COMMON_FEATURE_DOC,
        )
    }
}

evaluator!(Chi2Pvar, lcf::Chi2Pvar, StockTransformer::Identity);

macro_rules! color_two_band_feature {
    ($name:ident, $lcf_type:ident, $feature_doc:literal) => {
        #[derive(Serialize, Deserialize)]
        #[pyclass(extends = PyFeatureEvaluator, module = "light_curve.light_curve_ext")]
        pub struct $name {}

        impl_pickle_serialisation!($name);

        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (bands, *, transform=None))]
            fn __new__(bands: Bound<'_, PyAny>, transform: Option<Bound<PyAny>>) -> Res<(Self, PyFeatureEvaluator)> {
                if transform.is_some() {
                    return Err(Exception::NotImplementedError(
                        concat!(stringify!($name), " does not support transform").to_string(),
                    ));
                }
                let user_bands = parse_bands(&bands)?;
                if user_bands.len() != 2 {
                    return Err(Exception::ValueError(format!(
                        "bands must contain exactly 2 passbands, got {}",
                        user_bands.len()
                    )));
                }
                let mc_f32 = lcf::MultiColorFeature::$name(
                    lcf::multicolor::features::$lcf_type::new([
                        user_bands[0].clone(),
                        user_bands[1].clone(),
                    ]),
                );
                let mc_f64 = lcf::MultiColorFeature::$name(
                    lcf::multicolor::features::$lcf_type::new([
                        user_bands[0].clone(),
                        user_bands[1].clone(),
                    ]),
                );
                Ok((Self {}, PyFeatureEvaluator::multi_band(user_bands, mc_f32, mc_f64)))
            }

            #[staticmethod]
            fn __getnewargs__() -> ([&'static str; 2],) {
                (["g", "r"],)
            }

            #[classattr]
            fn __doc__() -> String {
                format!(
                    "{}\n\nParameters\n----------\nbands : list of two str\n    Two passband names.\n    The output is ``m[bands[0]] - m[bands[1]]``.\n\n{}",
                    $feature_doc,
                    COMMON_FEATURE_DOC,
                )
            }
        }
    };
}

color_two_band_feature!(
    ColorOfMaximum,
    ColorOfMaximum,
    "Difference of maximum magnitudes in two passbands.\n\n\
     Computes ``max(band[0]) - max(band[1])`` where the maximum is taken over each\n\
     passband independently. Note that maximum has mathematical meaning, not\n\
     the astronomical one (brighter objects have smaller magnitude)."
);

color_two_band_feature!(
    ColorOfMedian,
    ColorOfMedian,
    "Difference of median magnitudes in two passbands.\n\n\
     Computes ``median(band[0]) - median(band[1])`` where the median is taken\n\
     over each passband independently."
);

color_two_band_feature!(
    ColorOfMinimum,
    ColorOfMinimum,
    "Difference of minimum magnitudes in two passbands.\n\n\
     Computes ``min(band[0]) - min(band[1])`` where the minimum is taken over each\n\
     passband independently. Note that minimum has mathematical meaning, not\n\
     the astronomical one (fainter objects have larger magnitude)."
);

#[derive(Serialize, Deserialize)]
#[pyclass(extends = PyFeatureEvaluator, module = "light_curve.light_curve_ext")]
pub struct ColorSpread {}

impl_pickle_serialisation!(ColorSpread);

#[pymethods]
impl ColorSpread {
    #[new]
    #[pyo3(signature = (bands, *, transform=None))]
    fn __new__(
        bands: Bound<'_, PyAny>,
        transform: Option<Bound<PyAny>>,
    ) -> Res<(Self, PyFeatureEvaluator)> {
        if transform.is_some() {
            return Err(Exception::NotImplementedError(
                "ColorSpread does not support transform".to_string(),
            ));
        }
        let user_bands = parse_bands(&bands)?;
        if user_bands.len() < 2 {
            return Err(Exception::ValueError(format!(
                "bands must contain at least 2 passbands, got {}",
                user_bands.len()
            )));
        }
        let mc_f32 = lcf::MultiColorFeature::ColorSpread(
            lcf::multicolor::features::ColorSpread::new(user_bands.iter().cloned()),
        );
        let mc_f64 = lcf::MultiColorFeature::ColorSpread(
            lcf::multicolor::features::ColorSpread::new(user_bands.iter().cloned()),
        );
        Ok((
            Self {},
            PyFeatureEvaluator::multi_band(user_bands, mc_f32, mc_f64),
        ))
    }

    #[staticmethod]
    fn __getnewargs__() -> ([&'static str; 2],) {
        (["g", "r"],)
    }

    #[classattr]
    fn __doc__() -> &'static str {
        "Standard deviation of per-passband weighted mean magnitudes.\n\n\
         For each passband, the weighted mean magnitude is computed using inverse-variance\n\
         weights. ``ColorSpread`` is then the population standard deviation of these\n\
         per-band means. A large value indicates a large spread of mean brightnesses\n\
         across bands; zero means all bands have the same mean magnitude.\n\n\
         Parameters\n\
         ----------\n\
         bands : list of str\n\
             Two or more passband names.\n\
         \n\
         Attributes\n\
         ----------\n\
         names : list of str\n\
             Feature names\n\
         descriptions : list of str\n\
             Feature descriptions\n\
         bands : numpy.ndarray of str or None\n\
             Passband names for multiband mode, or None for single-band mode\n\
         \n\
         Methods\n\
         -------\n\
         __call__(self, t, m, sigma=None, band=None, *, fill_value=None, sorted=None, check=True, cast=False)\n\
             Extract features and return them as a numpy array\n\
         many(self, lcs, *, fill_value=None, sorted=None, check=True, cast=False, n_jobs=-1)\n\
             Extract features from multiple light curves in parallel"
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
    #[pyo3(signature = (quantile=lcf::InterPercentileRange::default_quantile(), *, transform = None, bands = None))]
    fn __new__(
        quantile: f32,
        transform: Option<Bound<PyAny>>,
        bands: Option<Bound<'_, PyAny>>,
    ) -> Res<PyClassInitializer<Self>> {
        let base = match bands {
            None => PyFeatureEvaluator::single_band(
                lcf::InterPercentileRange::new(quantile).into(),
                lcf::InterPercentileRange::new(quantile).into(),
                transform,
                Self::DEFAULT_TRANSFORMER,
            )?,
            Some(bands_py) => {
                let user_bands = parse_bands(&bands_py)?;
                let mc_f32 = lcf::MultiColorFeature::from_per_band_feature(
                    lcf::InterPercentileRange::new(quantile),
                    user_bands.clone(),
                );
                let mc_f64 = lcf::MultiColorFeature::from_per_band_feature(
                    lcf::InterPercentileRange::new(quantile),
                    user_bands.clone(),
                );
                PyFeatureEvaluator::multi_band(user_bands, mc_f32, mc_f64)
            }
        };
        Ok(PyClassInitializer::from(base).add_subclass(Self {}))
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
quantile : positive float, default {quantile_default:.2}
    Range is (100% × quantile, 100% × (1 - quantile))

{transform}
{bands}
{footer}"#,
            header = prepare_upstream_doc(lcf::InterPercentileRange::doc()),
            quantile_default = lcf::InterPercentileRange::default_quantile(),
            bands = BANDS_PARAMETER_DOC,
            transform = transform_parameter_doc(Self::DEFAULT_TRANSFORMER),
            footer = COMMON_FEATURE_DOC
        )
    }
}

evaluator!(Kurtosis, lcf::Kurtosis, StockTransformer::Arcsinh);

evaluator!(
    LaflerKinmanStringLength,
    lcf::LaflerKinmanStringLength,
    StockTransformer::Identity
);

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
        bands=None,
    ))]
    fn __new__(
        quantile_numerator: f32,
        quantile_denominator: f32,
        transform: Option<Bound<PyAny>>,
        bands: Option<Bound<'_, PyAny>>,
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
        let base = match bands {
            None => PyFeatureEvaluator::single_band(
                lcf::MagnitudePercentageRatio::new(quantile_numerator, quantile_denominator).into(),
                lcf::MagnitudePercentageRatio::new(quantile_numerator, quantile_denominator).into(),
                transform,
                Self::DEFAULT_TRANSFORMER,
            )?,
            Some(bands_py) => {
                let user_bands = parse_bands(&bands_py)?;
                let mc_f32 = lcf::MultiColorFeature::from_per_band_feature(
                    lcf::MagnitudePercentageRatio::new(quantile_numerator, quantile_denominator),
                    user_bands.clone(),
                );
                let mc_f64 = lcf::MultiColorFeature::from_per_band_feature(
                    lcf::MagnitudePercentageRatio::new(quantile_numerator, quantile_denominator),
                    user_bands.clone(),
                );
                PyFeatureEvaluator::multi_band(user_bands, mc_f32, mc_f64)
            }
        };
        Ok(PyClassInitializer::from(base).add_subclass(Self {}))
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
quantile_numerator : positive float, default {quantile_numerator_default:.2}
    Numerator inter-percentile range is (100% × q, 100% × (1 - q))

quantile_denominator : positive float, default {quantile_denominator_default:.2}
    Denominator inter-percentile range is (100% × q, 100% × (1 - q))

{transform}
{bands}
{footer}"#,
            header = prepare_upstream_doc(lcf::MagnitudePercentageRatio::doc()),
            quantile_numerator_default =
                lcf::MagnitudePercentageRatio::default_quantile_numerator(),
            quantile_denominator_default =
                lcf::MagnitudePercentageRatio::default_quantile_denominator(),
            transform = transform_parameter_doc(Self::DEFAULT_TRANSFORMER),
            bands = BANDS_PARAMETER_DOC,
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
    #[pyo3(signature = (quantile=lcf::MedianBufferRangePercentage::<f32>::default_quantile(), *, transform = None, bands = None))]
    fn __new__(
        quantile: f32,
        transform: Option<Bound<PyAny>>,
        bands: Option<Bound<'_, PyAny>>,
    ) -> Res<PyClassInitializer<Self>> {
        let base = match bands {
            None => PyFeatureEvaluator::single_band(
                lcf::MedianBufferRangePercentage::new(quantile).into(),
                lcf::MedianBufferRangePercentage::new(quantile).into(),
                transform,
                Self::DEFAULT_TRANSFORMER,
            )?,
            Some(bands_py) => {
                let user_bands = parse_bands(&bands_py)?;
                let mc_f32 = lcf::MultiColorFeature::from_per_band_feature(
                    lcf::MedianBufferRangePercentage::new(quantile),
                    user_bands.clone(),
                );
                let mc_f64 = lcf::MultiColorFeature::from_per_band_feature(
                    lcf::MedianBufferRangePercentage::new(quantile),
                    user_bands.clone(),
                );
                PyFeatureEvaluator::multi_band(user_bands, mc_f32, mc_f64)
            }
        };
        Ok(PyClassInitializer::from(base).add_subclass(Self {}))
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
quantile : positive float, default {quantile_default:.2}
    Relative range size

{transform}
{bands}
{footer}"#,
            header = prepare_upstream_doc(lcf::MedianBufferRangePercentage::<f64>::doc()),
            quantile_default = lcf::MedianBufferRangePercentage::<f64>::default_quantile(),
            transform = transform_parameter_doc(Self::DEFAULT_TRANSFORMER),
            bands = BANDS_PARAMETER_DOC,
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
    #[pyo3(signature = (quantile=lcf::PercentDifferenceMagnitudePercentile::default_quantile(), *, transform = None, bands = None))]
    fn __new__(
        quantile: f32,
        transform: Option<Bound<PyAny>>,
        bands: Option<Bound<'_, PyAny>>,
    ) -> Res<PyClassInitializer<Self>> {
        if !(0.0..0.5).contains(&quantile) {
            return Err(Exception::ValueError(
                "quantile must be between 0.0 and 0.5".to_string(),
            ));
        }
        let base = match bands {
            None => PyFeatureEvaluator::single_band(
                lcf::PercentDifferenceMagnitudePercentile::new(quantile).into(),
                lcf::PercentDifferenceMagnitudePercentile::new(quantile).into(),
                transform,
                Self::DEFAULT_TRANSFORMER,
            )?,
            Some(bands_py) => {
                let user_bands = parse_bands(&bands_py)?;
                let mc_f32 = lcf::MultiColorFeature::from_per_band_feature(
                    lcf::PercentDifferenceMagnitudePercentile::new(quantile),
                    user_bands.clone(),
                );
                let mc_f64 = lcf::MultiColorFeature::from_per_band_feature(
                    lcf::PercentDifferenceMagnitudePercentile::new(quantile),
                    user_bands.clone(),
                );
                PyFeatureEvaluator::multi_band(user_bands, mc_f32, mc_f64)
            }
        };
        Ok(PyClassInitializer::from(base).add_subclass(Self {}))
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
quantile : positive float, default {quantile_default:.2}
    Relative range size

{transform}
{bands}
{footer}"#,
            header = prepare_upstream_doc(lcf::PercentDifferenceMagnitudePercentile::doc()),
            quantile_default = lcf::PercentDifferenceMagnitudePercentile::default_quantile(),
            transform = transform_parameter_doc(Self::DEFAULT_TRANSFORMER),
            bands = BANDS_PARAMETER_DOC,
            footer = COMMON_FEATURE_DOC
        )
    }
}

type LcfPeriodogram<T> = lcf::Periodogram<T, Feature<T>>;
type McPeriodogram<T> = lcf::multicolor::features::MultiColorPeriodogram<Passband, T, Feature<T>>;
type McPeriodogramNorm = lcf::multicolor::features::MultiColorPeriodogramNormalisation;
type CreateEvalsResult = (
    LcfPeriodogram<f32>,
    LcfPeriodogram<f64>,
    Option<(McPeriodogram<f32>, McPeriodogram<f64>)>,
);

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

    fn parse_mc_normalization(s: &str) -> PyResult<McPeriodogramNorm> {
        match s {
            "count" => Ok(McPeriodogramNorm::Count),
            "chi2" => Ok(McPeriodogramNorm::Chi2),
            other => Err(PyValueError::new_err(format!(
                "multiband_normalization must be one of: 'count', 'chi2', got '{other}'"
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
        phase_features: Option<Bound<PyAny>>,
        normalization: PeriodogramNormalization,
        mc_params: Option<(Vec<Passband>, McPeriodogramNorm)>,
    ) -> PyResult<CreateEvalsResult> {
        let peaks_val = peaks.unwrap_or_else(LcfPeriodogram::<f32>::default_peaks);
        let mut eval_f32 = match peaks {
            Some(peaks) => lcf::Periodogram::new(peaks),
            None => lcf::Periodogram::default(),
        };
        let mut eval_f64 = match peaks {
            Some(peaks) => lcf::Periodogram::new(peaks),
            None => lcf::Periodogram::default(),
        };
        let mut mc = mc_params.as_ref().map(|(passband_set, mc_norm)| {
            let mc_f32: McPeriodogram<f32> =
                McPeriodogram::new(peaks_val, mc_norm.clone(), passband_set.iter().cloned());
            let mc_f64: McPeriodogram<f64> =
                McPeriodogram::new(peaks_val, mc_norm.clone(), passband_set.iter().cloned());
            (mc_f32, mc_f64)
        });

        if let Some(resolution) = resolution {
            eval_f32.set_freq_resolution(resolution);
            eval_f64.set_freq_resolution(resolution);
            if let Some((mc_f32, mc_f64)) = mc.as_mut() {
                mc_f32.set_freq_resolution(resolution);
                mc_f64.set_freq_resolution(resolution);
            }
        }
        if let Some(max_freq_factor) = max_freq_factor {
            eval_f32.set_max_freq_factor(max_freq_factor);
            eval_f64.set_max_freq_factor(max_freq_factor);
            if let Some((mc_f32, mc_f64)) = mc.as_mut() {
                mc_f32.set_max_freq_factor(max_freq_factor);
                mc_f64.set_max_freq_factor(max_freq_factor);
            }
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
            if let Some((mc_f32, mc_f64)) = mc.as_mut() {
                mc_f32.set_nyquist(nyquist_freq);
                mc_f64.set_nyquist(nyquist_freq);
            }
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
                if let Some((mc_f32, mc_f64)) = mc.as_mut() {
                    mc_f32.set_periodogram_algorithm(
                        lcf::PeriodogramPowerFft::<f32, lcf::periodogram::FftwFft<f32>>::new()
                            .into(),
                    );
                    mc_f64.set_periodogram_algorithm(
                        lcf::PeriodogramPowerFft::<f64, lcf::periodogram::FftwFft<f64>>::new()
                            .into(),
                    );
                }
            }
            #[cfg(not(feature = "mkl"))]
            {
                eval_f32.set_periodogram_algorithm(
                    lcf::PeriodogramPowerFft::<f32, lcf::periodogram::RustFft<f32>>::new().into(),
                );
                eval_f64.set_periodogram_algorithm(
                    lcf::PeriodogramPowerFft::<f64, lcf::periodogram::RustFft<f64>>::new().into(),
                );
                if let Some((mc_f32, mc_f64)) = mc.as_mut() {
                    mc_f32.set_periodogram_algorithm(
                        lcf::PeriodogramPowerFft::<f32, lcf::periodogram::RustFft<f32>>::new()
                            .into(),
                    );
                    mc_f64.set_periodogram_algorithm(
                        lcf::PeriodogramPowerFft::<f64, lcf::periodogram::RustFft<f64>>::new()
                            .into(),
                    );
                }
            }
        } else {
            eval_f32.set_periodogram_algorithm(lcf::PeriodogramPowerDirect {}.into());
            eval_f64.set_periodogram_algorithm(lcf::PeriodogramPowerDirect {}.into());
            if let Some((mc_f32, mc_f64)) = mc.as_mut() {
                mc_f32.set_periodogram_algorithm(lcf::PeriodogramPowerDirect {}.into());
                mc_f64.set_periodogram_algorithm(lcf::PeriodogramPowerDirect {}.into());
            }
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

            eval_f32.set_freq_grid(freq_grid_f32.clone());
            eval_f64.set_freq_grid(freq_grid_f64.clone());
            if let Some((mc_f32, mc_f64)) = mc.as_mut() {
                mc_f32.set_freq_grid(freq_grid_f32);
                mc_f64.set_freq_grid(freq_grid_f64);
            }
        }

        if let Some(features) = features {
            for x in features.try_iter()? {
                let py_feature = x?.cast::<PyFeatureEvaluator>()?.borrow();
                let (f32_eval, f64_eval) = match &py_feature.mode {
                    FeatureEvalMode::SingleBand {
                        feature_evaluator_f32,
                        feature_evaluator_f64,
                    } => (feature_evaluator_f32.clone(), feature_evaluator_f64.clone()),
                    FeatureEvalMode::MultiBand { .. } | FeatureEvalMode::Mixed { .. } => {
                        return Err(PyValueError::new_err(
                            "multiband features are not supported as Periodogram spectrum features",
                        ));
                    }
                };
                eval_f32.add_spectrum_feature(f32_eval.clone());
                eval_f64.add_spectrum_feature(f64_eval.clone());
                if let Some((mc_f32, mc_f64)) = mc.as_mut() {
                    mc_f32.add_spectrum_feature(f32_eval);
                    mc_f64.add_spectrum_feature(f64_eval);
                }
            }
        }

        if let Some(phase_features) = phase_features {
            // For multiband: register all passbands as phase bands before adding features
            if let Some((mc_f32, mc_f64)) = mc.as_mut() {
                let phase_bands: Vec<Passband> = mc_params.as_ref().unwrap().0.to_vec();
                mc_f32.set_phase_bands(phase_bands.clone());
                mc_f64.set_phase_bands(phase_bands);
            }
            for x in phase_features.try_iter()? {
                let py_feature = x?.cast::<PyFeatureEvaluator>()?.borrow();
                let (f32_eval, f64_eval) = match &py_feature.mode {
                    FeatureEvalMode::SingleBand {
                        feature_evaluator_f32,
                        feature_evaluator_f64,
                    } => (feature_evaluator_f32.clone(), feature_evaluator_f64.clone()),
                    FeatureEvalMode::MultiBand { .. } | FeatureEvalMode::Mixed { .. } => {
                        return Err(PyValueError::new_err(
                            "multiband features are not supported as Periodogram phase features",
                        ));
                    }
                };
                eval_f32.add_phase_feature(f32_eval.clone());
                eval_f64.add_phase_feature(f64_eval.clone());
                if let Some((mc_f32, mc_f64)) = mc.as_mut() {
                    mc_f32.add_phase_feature(f32_eval);
                    mc_f64.add_phase_feature(f64_eval);
                }
            }
        }

        eval_f32.set_normalization(normalization);
        eval_f64.set_normalization(normalization);

        Ok((eval_f32, eval_f64, mc))
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

    #[allow(clippy::too_many_arguments)]
    fn freq_power_mc_impl<'py, T>(
        mc: &McPeriodogram<T>,
        sorted_bands: &[Passband],
        band_input: &BandInput,
        py: Python<'py>,
        t: Arr<T>,
        m: Arr<T>,
        sigma: Option<Arr<T>>,
        band_py: &Bound<PyAny>,
    ) -> Res<(Bound<'py, PyUntypedArray>, Bound<'py, PyUntypedArray>)>
    where
        T: Float + numpy::Element,
    {
        let band_idx = band_array_to_indices(band_py, sorted_bands, band_input)?;
        // The multicolor periodogram weights per-band powers by w, so always gather it.
        let mut mcts = mcts_from_indices(
            t.as_array(),
            m.as_array(),
            sigma.as_ref().map(|s| s.as_array()),
            &band_idx,
            sorted_bands,
            true,
        );
        let (freq, power) = mc
            .freq_power(&mut mcts)
            .map_err(|e| Exception::ValueError(format!("{e:?}")))?;
        let freq = PyArray1::from_vec(py, freq.to_vec());
        let power = PyArray1::from_vec(py, power.to_vec());
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
        phase_features = None,
        normalization = "psd",
        transform = None,
        bands = None,
        multiband_normalization = "chi2",
    ))]
    fn __new__(
        peaks: Option<usize>,
        resolution: Option<f32>,
        max_freq_factor: Option<f32>,
        nyquist: Option<NyquistArgumentOfPeriodogram>,
        freqs: Option<Bound<PyAny>>,
        fast: Option<bool>,
        features: Option<Bound<PyAny>>,
        phase_features: Option<Bound<PyAny>>,
        normalization: &str,
        transform: Option<Bound<PyAny>>,
        bands: Option<Bound<'_, PyAny>>,
        multiband_normalization: &str,
    ) -> PyResult<(Self, PyFeatureEvaluator)> {
        if transform.is_some() {
            return Err(PyNotImplementedError::new_err(
                "transform is not supported by Periodogram, peak-related features are not transformed, but you still may apply transformation for the underlying features",
            ));
        }
        if bands.is_some() && normalization != "psd" {
            return Err(PyNotImplementedError::new_err(
                "normalization other than 'psd' is not supported for multiband Periodogram",
            ));
        }
        let normalization = Self::parse_normalization(normalization)?;
        let mc_params: Option<(Vec<Passband>, McPeriodogramNorm)> = match bands.as_ref() {
            None => None,
            Some(bands_py) => {
                let user_bands = parse_bands(bands_py)?;
                let mc_norm = Self::parse_mc_normalization(multiband_normalization)?;
                Some((user_bands, mc_norm))
            }
        };
        let (eval_f32, eval_f64, mc) = Self::create_evals(
            peaks,
            resolution,
            max_freq_factor,
            nyquist,
            freqs,
            fast,
            features,
            phase_features,
            normalization,
            mc_params.clone(),
        )?;
        let parent = match mc_params {
            None => PyFeatureEvaluator {
                mode: FeatureEvalMode::SingleBand {
                    feature_evaluator_f32: eval_f32.clone().into(),
                    feature_evaluator_f64: eval_f64.clone().into(),
                },
            },
            Some((bands, _)) => {
                let (mc_f32, mc_f64) = mc.expect("mc_params was Some so mc must be Some");
                PyFeatureEvaluator::multi_band(
                    bands,
                    lcf::MultiColorFeature::MultiColorPeriodogram(mc_f32),
                    lcf::MultiColorFeature::MultiColorPeriodogram(mc_f64),
                )
            }
        };
        Ok((
            Self {
                eval_f32: eval_f32.clone(),
                eval_f64: eval_f64.clone(),
            },
            parent,
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
    #[pyo3(signature = (t, m, sigma=None, band=None, *, cast=false))]
    fn freq_power<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        t: Bound<'py, PyAny>,
        m: Bound<'py, PyAny>,
        sigma: Option<Bound<'py, PyAny>>,
        band: Option<Bound<'py, PyAny>>,
        cast: bool,
    ) -> Res<(Bound<'py, PyUntypedArray>, Bound<'py, PyUntypedArray>)> {
        match &slf.as_super().mode {
            FeatureEvalMode::MultiBand {
                feature_evaluator_f32: mc_f32_any,
                feature_evaluator_f64: mc_f64_any,
                sorted_bands,
                band_input,
                ..
            } => {
                let (
                    lcf::MultiColorFeature::MultiColorPeriodogram(mc_f32),
                    lcf::MultiColorFeature::MultiColorPeriodogram(mc_f64),
                ) = (mc_f32_any.as_ref(), mc_f64_any.as_ref())
                else {
                    unreachable!("Periodogram in MultiBand mode always wraps McPeriodogram")
                };
                let band = band.ok_or_else(|| {
                    Exception::ValueError("band is required for multiband freq_power".to_string())
                })?;
                if let Some(sigma) = sigma {
                    dtype_dispatch!(
                        |t, m, sigma| Self::freq_power_mc_impl(mc_f32, sorted_bands, band_input, py, t, m, Some(sigma), &band),
                        |t, m, sigma| Self::freq_power_mc_impl(mc_f64, sorted_bands, band_input, py, t, m, Some(sigma), &band),
                        t, =m, =sigma; cast=cast
                    )
                } else {
                    dtype_dispatch!(
                        |t, m| Self::freq_power_mc_impl(mc_f32, sorted_bands, band_input, py, t, m, None, &band),
                        |t, m| Self::freq_power_mc_impl(mc_f64, sorted_bands, band_input, py, t, m, None, &band),
                        t, =m; cast=cast
                    )
                }
            }
            FeatureEvalMode::SingleBand { .. } | FeatureEvalMode::Mixed { .. } => {
                if sigma.is_some() || band.is_some() {
                    return Err(Exception::ValueError(
                        "sigma and band are only accepted by freq_power in multiband mode (bands= was set at construction)".to_string(),
                    ));
                }
                dtype_dispatch!(
                    |t, m| Self::freq_power_impl(&slf.eval_f32, py, t, m),
                    |t, m| Self::freq_power_impl(&slf.eval_f64, py, t, m),
                    t, =m; cast=cast
                )
            }
        }
    }

    #[classattr]
    fn __doc__() -> String {
        format!(
            r#"{intro}
Parameters
----------
peaks : int or None, default {default_peaks}
    Number of peaks to find

resolution : float or None, default {default_resolution}
    Resolution of frequency grid

max_freq_factor : float or None, default {default_max_freq_factor}
    Mulitplier for Nyquist frequency

nyquist : str or float or None, default '{default_nyquist}'
    Type of Nyquist frequency. Could be one of:

     - ``'average'``: "Average" Nyquist frequency
     - ``'median'``: Nyquist frequency is defined by median time interval
       between observations
     - float: Nyquist frequency is defined by given quantile of time
       intervals between observations

freqs : array-like or None, default None
    Explicit and fixed frequency grid (angular frequency, radians/time unit).
    If given, ``resolution``, ``max_freq_factor`` and ``nyquist`` are ignored.
    For ``fast=True`` the only supported type of the grid is
    ``np.linspace(0.0, max_freq, 2**k+1)``, where k is an integer.
    For ``fast=False`` any grid is accepted, but linear grids apply some
    computational optimisations.

fast : bool or None, default {default_fast}
    Use "Fast" (approximate and FFT-based) or direct periodogram algorithm

features : iterable or None, default None
    Features extracted from the periodogram power spectrum, treating it as a
    time-series (frequency as time, power as magnitude).
    ``None`` means no additional spectrum features.

phase_features : iterable or None, default None
    Features to extract from the light curve phase-folded at the best period.
    Phase runs from 0 to 1 with phase 0 at the magnitude minimum.
    Feature names are prefixed with ``period_folded_``.
    ``None`` means no phase features.

normalization : str, default 'psd'
    Normalization of the periodogram power. Affects ``power()``,
    ``freq_power()``, and feature extraction via ``__call__()``.
    Let P be the raw power and n the number of observations.
    Must be one of:

     - ``'psd'``: Raw power P, unnormalized. Consistent with
       ``scipy.signal.lombscargle(normalize=False)`` on variance-normalized
       data, but differs from astropy's ``'psd'`` convention
     - ``'standard'``: P_std = P * 2 / (n - 1), values in [0, 1].
       Matches astropy's ``'standard'`` normalization
     - ``'model'``: P_std / (1 - P_std), values in [0, inf).
       Matches astropy's ``'model'`` normalization
     - ``'log'``: -ln(1 - P_std), values in [0, inf).
       Matches astropy's ``'log'`` normalization

transform : None, default None
    Not supported for Periodogram. Peaks are not transformed, but you may
    apply transformation for the underlying features via their constructors

bands : list of str or None, optional
    Passband names for multiband mode. If provided, a multiband periodogram
    is evaluated across all passbands simultaneously using a joint frequency
    grid.

multiband_normalization : str, default 'chi2'
    How per-band power spectra are combined into the joint spectrum. Only used
    when ``bands`` is given. Must be one of:

     - ``'chi2'`` (default): each band is weighted by ``sum((m - m_mean)^2 / sigma^2)``.
       ``sigma`` is optional; unity weights are used when absent, which differs
       from ``'count'`` normalization and may not be meaningful.
     - ``'count'``: weight each passband by observation count, ignoring ``sigma``

{common}
freq_power(t, m, sigma=None, band=None, *, cast=False)
    Get periodogram as a pair of frequencies and power values.

    In **single-band** mode (``bands`` not set at construction) only ``t`` and
    ``m`` are accepted.

    In **multiband** mode (``bands`` set at construction) ``band`` is
    **required**. ``sigma`` is optional but recommended: it is used to weight
    each band by its chi-squared statistic when ``multiband_normalization='chi2'``
    (the default); unity weights are assumed when omitted.

    Parameters
    ----------
    t : np.ndarray of np.float32 or np.float64
        Time array

    m : np.ndarray of np.float32 or np.float64
        Magnitude (flux) array

    sigma : np.ndarray of np.float32 or np.float64, optional
        Photometric uncertainties. Used only for chi2-based band combination:
        each band is weighted by ``sum((m - m_mean)^2 / sigma^2)``.
        Unity weights are assumed when omitted, which changes the band weights
        but does not affect the per-observation Lomb-Scargle power computation.
        Ignored in single-band mode.

    band : array-like of str, required in multiband mode
        Passband label for each observation. Must be one of the bands given at
        construction.

    cast : bool, optional
        Cast inputs to np.ndarray objects of the same dtype

    Returns
    -------
    freq : np.ndarray of np.float32 or np.float64
        Frequency grid

    power : np.ndarray of np.float32 or np.float64
        Periodogram power (combined across bands in multiband mode)

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
            intro = prepare_upstream_doc(LcfPeriodogram::<f64>::doc()),
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
    #[pyo3(signature = (*, transform=None, bands=None))]
    fn __new__(
        transform: Option<Bound<PyAny>>,
        bands: Option<Bound<'_, PyAny>>,
    ) -> Res<(Self, PyFeatureEvaluator)> {
        if transform.is_some() {
            return Err(Exception::NotImplementedError(
                "OtsuSplit does not support transformations yet".to_string(),
            ));
        }
        let base = match bands {
            None => PyFeatureEvaluator {
                mode: FeatureEvalMode::SingleBand {
                    feature_evaluator_f32: lcf::OtsuSplit::new().into(),
                    feature_evaluator_f64: lcf::OtsuSplit::new().into(),
                },
            },
            Some(bands_py) => {
                let user_bands = parse_bands(&bands_py)?;
                let mc_f32 = lcf::MultiColorFeature::from_per_band_feature(
                    lcf::OtsuSplit::new(),
                    user_bands.clone(),
                );
                let mc_f64 = lcf::MultiColorFeature::from_per_band_feature(
                    lcf::OtsuSplit::new(),
                    user_bands.clone(),
                );
                PyFeatureEvaluator::multi_band(user_bands, mc_f32, mc_f64)
            }
        };
        Ok((Self {}, base))
    }

    #[staticmethod]
    fn threshold(m: Bound<PyAny>) -> Res<f64> {
        dtype_dispatch!({ Self::threshold_impl }(m))
    }

    #[classattr]
    fn __doc__() -> String {
        format!(
            "{}\nParameters\n----------\n{}\n{}",
            prepare_upstream_doc(lcf::OtsuSplit::doc()),
            BANDS_PARAMETER_DOC,
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
        #[derive(Deserialize)]
        struct MultiBandJsonF64 {
            bands: Vec<Passband>,
            feature: lcf::MultiColorFeature<Passband, f64>,
        }
        #[derive(Deserialize)]
        struct MultiBandJsonF32 {
            #[serde(rename = "bands")]
            _bands: serde::de::IgnoredAny,
            feature: lcf::MultiColorFeature<Passband, f32>,
        }

        // Detect multiband format by presence of the "bands" key.
        if let Ok(mb_f64) = serde_json::from_str::<MultiBandJsonF64>(&s) {
            let mb_f32 = serde_json::from_str::<MultiBandJsonF32>(&s).map_err(|err| {
                Exception::ValueError(format!(
                    "Cannot deserialize multiband feature from JSON: {err}"
                ))
            })?;
            return Ok((
                Self {},
                PyFeatureEvaluator::multi_band(mb_f64.bands, mb_f32.feature, mb_f64.feature),
            ));
        }

        let feature_evaluator_f32: Feature<f32> = serde_json::from_str(&s).map_err(|err| {
            Exception::ValueError(format!("Cannot deserialize feature from JSON: {err}"))
        })?;
        let feature_evaluator_f64: Feature<f64> = serde_json::from_str(&s).map_err(|err| {
            Exception::ValueError(format!("Cannot deserialize feature from JSON: {err}"))
        })?;

        Ok((
            Self {},
            PyFeatureEvaluator {
                mode: FeatureEvalMode::SingleBand {
                    feature_evaluator_f32,
                    feature_evaluator_f64,
                },
            },
        ))
    }
}
