use crate::errors::{Exception, Res};

use numpy::prelude::*;
use numpy::{Element, PyArray1, PyReadonlyArray1, PyUntypedArray, PyUntypedArrayMethods};
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

pub(crate) fn unknown_type_exception(name: &str, obj: Bound<PyAny>) -> Exception {
    let message = if let Ok(arr) = obj.downcast::<PyUntypedArray>() {
        let ndim = arr.ndim();
        if ndim != 1 {
            format!("'{name}' is a {ndim}-d array, only 1-d arrays are supported.")
        } else {
            let dtype = match arr.dtype().str() {
                Ok(s) => s,
                Err(err) => return err.into(),
            };
            format!("'{name}' has dtype {dtype}, but only float32 and float64 are supported.")
        }
    } else {
        let tp = match obj.get_type().name() {
            Ok(s) => s,
            Err(err) => return err.into(),
        };
        format!(
            "'{name}' has type '{tp}', float32 or float64 1-d numpy array was supported. Try to cast with np.asarray."
        )
    };
    Exception::TypeError(message)
}

pub(crate) fn extract_matched_array<'py, T>(
    y_name: &'static str,
    y: Bound<'py, PyAny>,
    x_name: &'static str,
    x: &Arr<'py, T>,
    check_size: bool,
) -> Res<Arr<'py, T>>
where
    T: Element + DType + 'py,
{
    if let Ok(y) = y.downcast::<PyArray1<T>>() {
        let y = y.readonly();
        if check_size {
            if y.len() == x.len() {
                Ok(y)
            } else {
                Err(Exception::ValueError(format!(
                    "Mismatched lengths: '{}': {}, '{}': {}",
                    x_name,
                    x.len(),
                    y_name,
                    y.len(),
                )))
            }
        } else {
            Ok(y)
        }
    } else {
        let error_message = if let Ok(y_arr) = y.downcast::<PyUntypedArray>() {
            if y_arr.ndim() != 1 {
                format!(
                    "'{}' is a {}-d array, only 1-d arrays are supported.",
                    y_name,
                    y_arr.ndim()
                )
            } else {
                format!(
                    "Mismatched dtypes: '{}': {}, '{}': {}",
                    x_name,
                    x.dtype().str()?,
                    y_name,
                    y_arr.dtype().str()?
                )
            }
        } else {
            format!("'{y_name}' must be a numpy array of the same shape and dtype as '{x_name}', '{x_name}' has type 'np.ndarray[{x_dtype}]', '{y_name}' has type '{y_type}')", y_type=y.get_type().name()?, x_dtype=T::dtype_name())
        };
        Err(Exception::TypeError(error_message))
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
            Err(crate::np_array::unknown_type_exception(stringify!($first_arg), $first_arg.clone()))
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
            Err(crate::np_array::unknown_type_exception(stringify!($first_arg), $first_arg.clone()))
        }
    }};
}
