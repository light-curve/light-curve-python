use crate::errors::{Exception, Res};

use itertools::{izip, Either};
use numpy::prelude::*;
use numpy::{
    AllowTypeChange, PyArray1, PyArrayLike1, PyReadonlyArray1, PyUntypedArray,
    PyUntypedArrayMethods,
};
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

pub(crate) fn unknown_type_exception(name: &str, obj: &Bound<PyAny>) -> Exception {
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

fn cast_fail_reason(
    idx: usize,
    names: &'static [&'static str],
    objects: &[&Bound<PyAny>],
    good_f32_arrays: &[PyReadonlyArray1<f32>],
    good_f64_arrays: &[PyReadonlyArray1<f64>],
) -> Exception {
    let first_name = names.first().expect("Empty names slice");
    let fist_obj = objects.first().expect("Empty objects slice");

    // If the very first argument downcast
    if idx == 0 {
        return unknown_type_exception(first_name, fist_obj);
    }

    let (first_arr, first_dtype_name) = if good_f32_arrays.is_empty() {
        (
            good_f64_arrays
                .first()
                .expect("Empty good_f64_arrays slice")
                .as_untyped(),
            f64::dtype_name(),
        )
    } else {
        (
            good_f32_arrays
                .first()
                .expect("Empty good_f32_arrays slice")
                .as_untyped(),
            f32::dtype_name(),
        )
    };

    let fail_name = names.get(idx).expect("idx is out of bounds of names slice");
    let fail_obj = objects
        .get(idx)
        .expect("idx is out of bounds of names slice");

    let error_message = if let Ok(fail_arr) = fail_obj.downcast::<PyUntypedArray>() {
        if fail_arr.ndim() != 1 {
            format!(
                "'{}' is a {}-d array, only 1-d arrays are supported.",
                fail_name,
                fail_arr.ndim()
            )
        } else {
            let first_arr_dtype_str = match first_arr.dtype().str() {
                Ok(s) => s,
                Err(err) => return err.into(),
            };
            let fail_arr_dtype_str = match fail_arr.dtype().str() {
                Ok(s) => s,
                Err(err) => return err.into(),
            };
            format!(
                "Mismatched dtypes: '{first_name}': {first_arr_dtype_str}, '{fail_name}': {fail_arr_dtype_str}",
            )
        }
    } else {
        let fail_obj_type_name = match fail_obj.get_type().name() {
            Ok(s) => s,
            Err(err) => return err.into(),
        };
        format!(
            "'{fail_name}' must be a numpy array of the same shape and dtype as '{first_name}', '{first_name}' has type 'np.ndarray[{first_dtype_name}]', '{fail_name}' has type '{fail_obj_type_name}')",
        )
    };
    Exception::TypeError(error_message)
}

pub(crate) enum GenericVecOfArrays<'py> {
    F32(Vec<PyReadonlyArray1<'py, f32>>),
    F64(Vec<PyReadonlyArray1<'py, f64>>),
}

impl<'py> GenericVecOfArrays<'py> {
    fn first_len(&self) -> Option<usize> {
        match self {
            Self::F32(v) => v.first().map(|arr| arr.len()),
            Self::F64(v) => v.first().map(|arr| arr.len()),
        }
    }

    fn iter_len(&'py self) -> impl Iterator<Item = usize> + 'py {
        match self {
            Self::F32(v) => Either::Left(v.iter().map(|arr| arr.len())),
            Self::F64(v) => Either::Right(v.iter().map(|arr| arr.len())),
        }
    }
}

fn try_downcast_objects_to_f32_arrays<'py>(
    objects: &[&Bound<'py, PyAny>],
) -> Vec<PyReadonlyArray1<'py, f32>> {
    objects
        .iter()
        .map(|&obj| obj.downcast::<PyArray1<f32>>())
        .take_while(|result| result.is_ok())
        // It is safe to unwrap, because we have just checked that it is Ok
        .map(|result| result.unwrap().readonly())
        .collect()
}

fn try_downcast_to_f64_array<'py>(obj: &Bound<'py, PyAny>) -> Option<PyReadonlyArray1<'py, f64>> {
    if let Ok(py_array) = obj.downcast::<PyArray1<f64>>() {
        Some(py_array.readonly())
    } else if let Ok(as_array) = PyArrayLike1::<f64, AllowTypeChange>::extract_bound(obj) {
        Some(as_array.readonly())
    } else {
        None
    }
}

fn downcast_objects<'py>(
    names: &'static [&'static str],
    objects: &[&Bound<'py, PyAny>],
) -> Res<GenericVecOfArrays<'py>> {
    let f32_arrays = try_downcast_objects_to_f32_arrays(objects);
    if f32_arrays.len() == names.len() {
        Ok(GenericVecOfArrays::F32(f32_arrays))
    } else {
        let mut result = Vec::with_capacity(names.len());
        for f32_arr in f32_arrays.iter() {
            let f64_arr = f32_arr.cast::<f64>(false)?;
            result.push(f64_arr.readonly());
        }
        for (idx, &obj) in objects.iter().enumerate().skip(result.len()) {
            let py_ro_array = try_downcast_to_f64_array(obj)
                .ok_or_else(|| cast_fail_reason(idx, names, objects, &f32_arrays, &result))?;
            result.push(py_ro_array);
        }
        Ok(GenericVecOfArrays::F64(result))
    }
}

pub(crate) fn downcast_and_validate<'py>(
    names: &'static [&'static str],
    objects: &[&Bound<'py, PyAny>],
    check_size: &[bool],
) -> Res<GenericVecOfArrays<'py>> {
    assert_eq!(names.len(), objects.len());

    let arrays = downcast_objects(names, objects)?;
    let first_array_len = match arrays.first_len() {
        Some(len) => len,
        None => return Ok(arrays),
    };
    // We checked that 1) names size matches objects size, 2) objects is not empty
    let first_name = names[0];

    for (&name, length, &check) in izip!(&names[1..], arrays.iter_len().skip(1), check_size) {
        if check && length != first_array_len {
            return Err(Exception::ValueError(format!(
                "Mismatched lengths: '{}': {}, '{}': {}",
                first_name, first_array_len, name, length,
            )));
        }
    }
    Ok(arrays)
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
    // @call variants are for the internal usage only
    (@call $func:expr, $arrays:ident, $_arg1:expr,) => {{
        let func = $func;
        func($arrays[0].clone())
    }};
    (@call $func:expr, $arrays:ident, $_arg1:expr, $_arg2:expr,) => {{
        let func = $func;
        func($arrays[0].clone(), $arrays[1].clone())
    }};
    (@call $func:expr, $arrays:ident, $_arg1:expr, $_arg2:expr, $_arg3:expr,) => {{
        let func = $func;
        func($arrays[0].clone(), $arrays[1].clone(), $arrays[2].clone())
    }};
    ($func:tt ($first_arg:expr $(,$eq:tt $arg:expr)* $(,)?)) => {
        dtype_dispatch!($func, $func, $first_arg $(,$eq $arg)*)
    };
    ($f32:expr, $f64:expr, $first_arg:expr $(,$eq:tt $arg:expr)* $(,)?) => {{
        let names = &[stringify!($first_arg), $(stringify!($arg), )*];
        let objects = &[&$first_arg, $(&$arg, )*];
        let check_size = &[ $(_distinguish_eq_symbol!($eq), )* ];

        let generic_arrays = crate::np_array::downcast_and_validate(names, objects, check_size)?;
        match generic_arrays {
            crate::np_array::GenericVecOfArrays::F32(arrays) => {
                dtype_dispatch!(@call $f32, arrays, $first_arg, $($arg,)*)
            }
            crate::np_array::GenericVecOfArrays::F64(arrays) => {
                dtype_dispatch!(@call $f64, arrays, $first_arg, $($arg,)*)
            }
        }
    }};
}
