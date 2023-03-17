use crate::errors::{Exception, Res};

use numpy::{Element, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

pub(crate) type Arr<'a, T> = PyReadonlyArray1<'a, T>;

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
    check_size: bool,
) -> Res<Arr<'py, T>>
where
    T: Element + DType,
{
    if let Ok(y) = y.downcast::<PyArray1<T>>() {
        let y = y.readonly();
        if check_size {
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
            Ok(y)
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

macro_rules! _distinguish_eq_symbol {
    (=) => {
        true
    };
    (!=) => {
        false
    };
}

macro_rules! dtype_dispatch {
    ($func: tt ($first_arg:expr $(,$eq:tt $arg:expr)* $(,)?)) => {
        dtype_dispatch!($func, $func, $first_arg $(,$eq $arg)*)
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
    ($f32:expr, $f64:expr, $first_arg:expr $(,$eq:tt $arg:expr)+ $(,)?) => {{
        let x_name = stringify!($first_arg);
        if let Ok(x32) = $first_arg.downcast::<numpy::PyArray1<f32>>() {
            let x32 = x32.readonly();
            let f32 = $f32;
            f32(x32.clone(), $(crate::np_array::extract_matched_array(stringify!($arg), $arg, x_name, &x32, _distinguish_eq_symbol!($eq))?,)*)
        } else if let Ok(x64) = $first_arg.downcast::<numpy::PyArray1<f64>>() {
            let x64 = x64.readonly();
            let f64 = $f64;
            f64(x64.clone(), $(crate::np_array::extract_matched_array(stringify!($arg), $arg, x_name, &x64, _distinguish_eq_symbol!($eq))?,)*)
        } else {
            Err(crate::errors::Exception::TypeError("Unsupported dtype".into()).into())
        }
    }};
}
