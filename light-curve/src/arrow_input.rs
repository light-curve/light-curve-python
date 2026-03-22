use arrow_array::types::{ArrowPrimitiveType, Float32Type, Float64Type};
use arrow_buffer::ArrowNativeType;
use arrow_schema::DataType;
use pyo3::prelude::*;
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

/// Specifies which struct fields to use as t, m, and optionally sigma.
/// PyO3 will try `Vec<String>` first, then `Vec<usize>`. Mixed lists are
/// rejected automatically because neither variant will match.
#[derive(FromPyObject)]
pub(crate) enum PyArrowFields {
    Names(Vec<String>),
    Indices(Vec<usize>),
}

pub(crate) struct ArrowLcsSchema {
    pub dtype: ArrowDtype,
    pub t_idx: usize,
    pub m_idx: usize,
    pub sigma_idx: Option<usize>,
    pub list_type: ArrowListType,
}

fn resolve_by_name(fields: &arrow_schema::Fields, name: &str, role: &str) -> Res<usize> {
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
            it appears {} times in the struct; use an index instead",
            matches.len()
        ))),
    }
}

fn resolve_by_index(fields: &arrow_schema::Fields, idx: usize, role: &str) -> Res<usize> {
    if idx >= fields.len() {
        Err(Exception::ValueError(format!(
            "arrow_fields: index {idx} for {role} is out of range, struct has {} fields",
            fields.len()
        )))
    } else {
        Ok(idx)
    }
}

fn resolve_tms(
    struct_fields: &arrow_schema::Fields,
    arrow_fields: &PyArrowFields,
) -> Res<(usize, usize, Option<usize>)> {
    match arrow_fields {
        PyArrowFields::Names(names) => match names.as_slice() {
            [t, m] => {
                let ti = resolve_by_name(struct_fields, t, "t")?;
                let mi = resolve_by_name(struct_fields, m, "m")?;
                if ti == mi {
                    return Err(Exception::ValueError(
                        "arrow_fields: t and m must refer to different fields".to_string(),
                    ));
                }
                Ok((ti, mi, None))
            }
            [t, m, s] => {
                let ti = resolve_by_name(struct_fields, t, "t")?;
                let mi = resolve_by_name(struct_fields, m, "m")?;
                let si = resolve_by_name(struct_fields, s, "sigma")?;
                if ti == mi || ti == si || mi == si {
                    return Err(Exception::ValueError(
                        "arrow_fields: t, m, and sigma must refer to different fields".to_string(),
                    ));
                }
                Ok((ti, mi, Some(si)))
            }
            other => Err(Exception::ValueError(format!(
                "arrow_fields must have 2 (t, m) or 3 (t, m, sigma) elements, got {}",
                other.len()
            ))),
        },
        PyArrowFields::Indices(indices) => match indices.as_slice() {
            [t, m] => {
                let ti = resolve_by_index(struct_fields, *t, "t")?;
                let mi = resolve_by_index(struct_fields, *m, "m")?;
                if ti == mi {
                    return Err(Exception::ValueError(
                        "arrow_fields: t and m must refer to different fields".to_string(),
                    ));
                }
                Ok((ti, mi, None))
            }
            [t, m, s] => {
                let ti = resolve_by_index(struct_fields, *t, "t")?;
                let mi = resolve_by_index(struct_fields, *m, "m")?;
                let si = resolve_by_index(struct_fields, *s, "sigma")?;
                if ti == mi || ti == si || mi == si {
                    return Err(Exception::ValueError(
                        "arrow_fields: t, m, and sigma must refer to different fields".to_string(),
                    ));
                }
                Ok((ti, mi, Some(si)))
            }
            other => Err(Exception::ValueError(format!(
                "arrow_fields must have 2 (t, m) or 3 (t, m, sigma) elements, got {}",
                other.len()
            ))),
        },
    }
}

/// Validate that the chunked array has type `List<Struct<...>>` and return schema info.
///
/// `arrow_fields` specifies which struct fields to use as t, m, and optionally sigma,
/// either by name or by zero-based index. The struct may have any number of fields.
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
    let (t_idx, m_idx, sigma_idx) = resolve_tms(struct_fields, arrow_fields)?;

    // 4. Check that the selected fields all share the same float dtype
    let selected_indices: Vec<usize> = [Some(t_idx), Some(m_idx), sigma_idx]
        .into_iter()
        .flatten()
        .collect();
    let first_dt = struct_fields[selected_indices[0]].data_type();
    for &idx in &selected_indices[1..] {
        let dt = struct_fields[idx].data_type();
        if dt != first_dt {
            let role = if idx == m_idx { "m" } else { "sigma" };
            return Err(Exception::TypeError(format!(
                "All selected struct fields must have the same dtype, \
                t (field {t_idx}) is {first_dt:?} but {role} (field {idx}) is {dt:?}"
            )));
        }
    }

    let dtype = match first_dt {
        DataType::Float32 => ArrowDtype::F32,
        DataType::Float64 => ArrowDtype::F64,
        other => {
            return Err(Exception::TypeError(format!(
                "Struct fields must be Float32 or Float64, got {other:?}"
            )));
        }
    };

    Ok(ArrowLcsSchema {
        dtype,
        t_idx,
        m_idx,
        sigma_idx,
        list_type,
    })
}
