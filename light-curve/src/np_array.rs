use crate::errors::{Exception, Res};

use numpy::{Element, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::convert::TryFrom;

pub(crate) type Arr<'a, T> = PyReadonlyArray1<'a, T>;

#[derive(FromPyObject)]
pub(crate) enum GenericFloatArray1<'a> {
    #[pyo3(transparent, annotation = "np.ndarray[float32]")]
    Float32(Arr<'a, f32>),
    #[pyo3(transparent, annotation = "np.ndarray[float64]")]
    Float64(Arr<'a, f64>),
}

impl<'a> TryFrom<GenericFloatArray1<'a>> for Arr<'a, f32> {
    type Error = ();

    fn try_from(value: GenericFloatArray1<'a>) -> Result<Self, Self::Error> {
        match value {
            GenericFloatArray1::Float32(a) => Ok(a),
            GenericFloatArray1::Float64(_) => Err(()),
        }
    }
}

impl<'a> TryFrom<GenericFloatArray1<'a>> for Arr<'a, f64> {
    type Error = ();

    fn try_from(value: GenericFloatArray1<'a>) -> Result<Self, Self::Error> {
        match value {
            GenericFloatArray1::Float32(_) => Err(()),
            GenericFloatArray1::Float64(a) => Ok(a),
        }
    }
}

pub(crate) trait DType {
    fn dtype_name() -> &'static str;
}

impl DType for f32 {
    fn dtype_name() -> &'static str {
        "float32"
    }
}

impl DType for f64 {
    fn dtype_name() -> &'static str {
        "float64"
    }
}

pub(crate) fn extract_matched_array<'py, T>(
    y_name: &'static str,
    y: &'py PyAny,
    x_name: &'static str,
    x: &Arr<'py, T>,
) -> Res<Arr<'py, T>>
where
    T: Element + DType,
{
    if let Ok(y) = y.downcast::<PyArray1<T>>() {
        let y = y.readonly();
        if y.len() == x.len() {
            Ok(y)
        } else {
            Err(Exception::ValueError(format!(
                "Mismatched length ({}: {}, {}: {})",
                y_name,
                y.len(),
                x_name,
                x.len(),
            )))
        }
    } else {
        let y_type = y
            .get_type()
            .name()
            .map(|name| {
                if name == "ndarray" {
                    format!(
                        "ndarray[{}]",
                        y.getattr("dtype")
                            .map(|dtype| dtype
                                .getattr("name")
                                .map(|p| p.to_string())
                                .unwrap_or("unknown".into()))
                            .unwrap_or("unknown".into())
                    )
                } else {
                    name.to_string()
                }
            })
            .unwrap_or("unknown".into());
        Err(Exception::TypeError(format!(
            "Mismatched types ({}: np.ndarray[{}], {}: {})",
            x_name,
            T::dtype_name(),
            y_name,
            y_type
        )))
    }
}

macro_rules! dtype_dispatch {
    ($func: tt ($first_arg:expr $(,$arg:expr)* $(,)?)) => {
        dtype_dispatch!($func, $func, $first_arg $(,$arg)*)
    };
    ($f32:expr, $f64:expr, $first_arg:expr $(,)?) => {{
        if let Ok(x32) = $first_arg.downcast::<numpy::PyArray1<f32>>() {
            let x32 = x32.readonly();
            let f32 = $f32;
            f32(x32)
        } else if let Ok(x64) = $first_arg.downcast::<numpy::PyArray1<f64>>() {
            let x64 = x64.readonly();
            let f64 = $f64;
            f64(x64)
        } else {
            Err(crate::errors::Exception::TypeError("Unsupported dtype".into()).into())
        }
    }};
    ($f32:expr, $f64:expr, $first_arg:expr $(,$arg:expr)+ $(,)?) => {{
        let x_name = stringify!($first_arg);
        if let Ok(x32) = $first_arg.downcast::<numpy::PyArray1<f32>>() {
            let x32 = x32.readonly();
            let f32 = $f32;
            f32(x32.clone(), $(crate::np_array::extract_matched_array(stringify!($arg), $arg, x_name, &x32)?,)*)
        } else if let Ok(x64) = $first_arg.downcast::<numpy::PyArray1<f64>>() {
            let x64 = x64.readonly();
            let f64 = $f64;
            f64(x64.clone(), $(crate::np_array::extract_matched_array(stringify!($arg), $arg, x_name, &x64)?,)*)
        } else {
            Err(crate::errors::Exception::TypeError("Unsupported dtype".into()).into())
        }
    }};
}
