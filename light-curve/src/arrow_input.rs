use arrow_array::types::{ArrowPrimitiveType, Float32Type, Float64Type};
use arrow_buffer::ArrowNativeType;
use arrow_schema::DataType;
use pyo3::exceptions::{PyKeyError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3_arrow::PyChunkedArray;

use crate::errors::{Exception, Res};

/// Bridge trait: maps f32→Float32Type, f64→Float64Type for downcasting arrow arrays.
pub(crate) trait ArrowFloat:
    light_curve_feature::Float + numpy::Element + ArrowNativeType
{
    type ArrowType: ArrowPrimitiveType<Native = Self>;
}

impl ArrowFloat for f32 {
    type ArrowType = Float32Type;
}

impl ArrowFloat for f64 {
    type ArrowType = Float64Type;
}

pub(crate) enum ArrowDtype {
    F32,
    F64,
}

pub(crate) enum ArrowListType {
    List,
    LargeList,
}

/// Whether the band column uses Utf8/LargeUtf8/Utf8View (string) or a signed/unsigned integer type.
#[derive(Clone)]
pub(crate) enum ArrowBandType {
    Utf8,
    LargeUtf8,
    Utf8View,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
}

/// A column reference: either a field name or a zero-based index.
pub(crate) enum FieldRef {
    Name(String),
    Index(usize),
}

impl<'a, 'py> FromPyObject<'a, 'py> for FieldRef {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(s) = obj.extract::<String>() {
            return Ok(FieldRef::Name(s));
        }
        if let Ok(i) = obj.extract::<usize>() {
            return Ok(FieldRef::Index(i));
        }
        Err(PyTypeError::new_err(
            "each field reference must be a column name (str) or a zero-based index (int)",
        ))
    }
}

/// Specifies which struct fields to use as t, m, and optionally sigma and band.
///
/// Pass as a dict, e.g. `{"t": "time", "m": "flux", "sigma": "fluxerr", "band": "passband"}`.
/// Values can be field names (str) or zero-based indices (int).
pub(crate) struct PyArrowFields {
    pub t: FieldRef,
    pub m: FieldRef,
    pub sigma: Option<FieldRef>,
    pub band: Option<FieldRef>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyArrowFields {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let dict = if let Ok(d) = obj.cast::<PyDict>() {
            d
        } else if obj.is_instance_of::<PyList>() {
            return Err(PyTypeError::new_err(
                "arrow_fields no longer accepts a list; pass a dict instead, e.g. \
                {\"t\": \"time\", \"m\": \"mag\"} with optional \"sigma\" and \"band\" keys",
            ));
        } else {
            return Err(PyTypeError::new_err(
                "arrow_fields must be a dict, e.g. {\"t\": \"time\", \"m\": \"mag\"}",
            ));
        };

        let get = |key: &str| -> PyResult<Option<FieldRef>> {
            match dict.get_item(key)? {
                Some(v) => Ok(Some(v.extract::<FieldRef>()?)),
                None => Ok(None),
            }
        };

        let t = get("t")?.ok_or_else(|| PyKeyError::new_err("arrow_fields must contain \"t\""))?;
        let m = get("m")?.ok_or_else(|| PyKeyError::new_err("arrow_fields must contain \"m\""))?;
        let sigma = get("sigma")?;
        let band = get("band")?;

        Ok(PyArrowFields { t, m, sigma, band })
    }
}

pub(crate) struct ArrowLcsSchema {
    pub dtype: ArrowDtype,
    pub t_idx: usize,
    pub m_idx: usize,
    pub sigma_idx: Option<usize>,
    pub band_idx: Option<usize>,
    pub band_type: Option<ArrowBandType>,
    pub list_type: ArrowListType,
}

fn resolve_field(fields: &arrow_schema::Fields, field_ref: &FieldRef, role: &str) -> Res<usize> {
    match field_ref {
        FieldRef::Name(name) => {
            let matches: Vec<usize> = fields
                .iter()
                .enumerate()
                .filter_map(|(i, f)| (f.name() == name).then_some(i))
                .collect();
            match matches.len() {
                0 => Err(Exception::ValueError(format!(
                    "arrow_fields: field name {name:?} for {role} not found in struct fields {:?}",
                    fields.iter().map(|f| f.name()).collect::<Vec<_>>()
                ))),
                1 => Ok(matches[0]),
                _ => Err(Exception::ValueError(format!(
                    "arrow_fields: field name {name:?} for {role} is ambiguous — \
                    it appears {} times; use an index instead",
                    matches.len()
                ))),
            }
        }
        FieldRef::Index(idx) => {
            if *idx >= fields.len() {
                Err(Exception::ValueError(format!(
                    "arrow_fields: index {idx} for {role} is out of range, struct has {} fields",
                    fields.len()
                )))
            } else {
                Ok(*idx)
            }
        }
    }
}

/// Validate that the chunked array has type `List<Struct<...>>` and return schema info.
///
/// `arrow_fields` is a dict specifying which struct fields map to t, m, sigma (optional),
/// and band (optional), by name or zero-based index.
pub(crate) fn validate_arrow_lcs(
    chunked: &PyChunkedArray,
    arrow_fields: &PyArrowFields,
) -> Res<ArrowLcsSchema> {
    let data_type = chunked.data_type();

    // 1. Check outer type is List or LargeList
    let (inner_field, list_type) = match data_type {
        DataType::List(field) => (field, ArrowListType::List),
        DataType::LargeList(field) => (field, ArrowListType::LargeList),
        other => {
            return Err(Exception::TypeError(format!(
                "Arrow input must be a List or LargeList array, got {other:?}"
            )));
        }
    };

    // 2. Check inner type is Struct
    let struct_fields = match inner_field.data_type() {
        DataType::Struct(fields) => fields,
        other => {
            return Err(Exception::TypeError(format!(
                "Arrow List elements must be Struct, got {other:?}"
            )));
        }
    };

    // 3. Resolve field indices
    let t_idx = resolve_field(struct_fields, &arrow_fields.t, "t")?;
    let m_idx = resolve_field(struct_fields, &arrow_fields.m, "m")?;
    let sigma_idx = arrow_fields
        .sigma
        .as_ref()
        .map(|s| resolve_field(struct_fields, s, "sigma"))
        .transpose()?;
    let (band_idx, band_type) = match &arrow_fields.band {
        None => (None, None),
        Some(band_ref) => {
            let idx = resolve_field(struct_fields, band_ref, "band")?;
            let band_type = match struct_fields[idx].data_type() {
                DataType::Utf8 => ArrowBandType::Utf8,
                DataType::LargeUtf8 => ArrowBandType::LargeUtf8,
                DataType::Utf8View => ArrowBandType::Utf8View,
                DataType::Int8 => ArrowBandType::Int8,
                DataType::Int16 => ArrowBandType::Int16,
                DataType::Int32 => ArrowBandType::Int32,
                DataType::Int64 => ArrowBandType::Int64,
                DataType::UInt8 => ArrowBandType::UInt8,
                DataType::UInt16 => ArrowBandType::UInt16,
                DataType::UInt32 => ArrowBandType::UInt32,
                DataType::UInt64 => ArrowBandType::UInt64,
                other => {
                    return Err(Exception::TypeError(format!(
                        "band field must be a string (Utf8, LargeUtf8, Utf8View) \
                        or integer type, got {other:?}"
                    )));
                }
            };
            (Some(idx), Some(band_type))
        }
    };

    // 4. Check uniqueness
    let numeric_indices: Vec<(usize, &str)> = [
        Some((t_idx, "t")),
        Some((m_idx, "m")),
        sigma_idx.map(|i| (i, "sigma")),
    ]
    .into_iter()
    .flatten()
    .collect();
    for i in 0..numeric_indices.len() {
        for j in (i + 1)..numeric_indices.len() {
            if numeric_indices[i].0 == numeric_indices[j].0 {
                return Err(Exception::ValueError(format!(
                    "arrow_fields: {} and {} must refer to different fields",
                    numeric_indices[i].1, numeric_indices[j].1
                )));
            }
        }
    }

    // 5. Check that t/m/sigma all share the same float dtype
    let numeric_idx_list: Vec<usize> = [Some(t_idx), Some(m_idx), sigma_idx]
        .into_iter()
        .flatten()
        .collect();
    let first_dt = struct_fields[numeric_idx_list[0]].data_type();
    for &idx in &numeric_idx_list[1..] {
        let dt = struct_fields[idx].data_type();
        if dt != first_dt {
            let role = if idx == m_idx { "m" } else { "sigma" };
            return Err(Exception::TypeError(format!(
                "All numeric struct fields must have the same dtype; \
                t (field {t_idx}) is {first_dt:?} but {role} (field {idx}) is {dt:?}"
            )));
        }
    }

    let dtype = match first_dt {
        DataType::Float32 => ArrowDtype::F32,
        DataType::Float64 => ArrowDtype::F64,
        other => {
            return Err(Exception::TypeError(format!(
                "Struct fields for t/m/sigma must be Float32 or Float64, got {other:?}"
            )));
        }
    };

    Ok(ArrowLcsSchema {
        dtype,
        t_idx,
        m_idx,
        sigma_idx,
        band_idx,
        band_type,
        list_type,
    })
}
