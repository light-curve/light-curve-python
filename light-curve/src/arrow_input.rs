use arrow_array::types::{ArrowPrimitiveType, Float32Type, Float64Type};
use arrow_buffer::ArrowNativeType;
use arrow_schema::DataType;
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

pub(crate) struct ArrowLcsSchema {
    pub dtype: ArrowDtype,
    pub has_sigma: bool,
    pub list_type: ArrowListType,
}

/// Validate that the chunked array has type `List<Struct<f, f[, f]>>` and return schema info.
pub(crate) fn validate_arrow_lcs(chunked: &PyChunkedArray) -> Res<ArrowLcsSchema> {
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

    // 3. Check field count (2 or 3)
    let has_sigma = match struct_fields.len() {
        2 => false,
        3 => true,
        n => {
            return Err(Exception::ValueError(format!(
                "Struct must have 2 (t, m) or 3 (t, m, sigma) fields, got {n}"
            )));
        }
    };

    // 4. Check all fields have the same float dtype
    let first_dt = struct_fields[0].data_type();
    for (i, field) in struct_fields.iter().enumerate().skip(1) {
        if field.data_type() != first_dt {
            return Err(Exception::TypeError(format!(
                "All struct fields must have the same dtype, field 0 is {first_dt:?} but field {i} is {:?}",
                field.data_type()
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
        has_sigma,
        list_type,
    })
}
