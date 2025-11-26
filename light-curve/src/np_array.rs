use crate::errors::{Exception, Res};

use numpy::prelude::*;
use numpy::{
    AllowTypeChange, PyArray1, PyArrayLike1, PyReadonlyArray1, PyUntypedArray,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use unarray::UnarrayArrayExt;

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

fn cast_fail_reason<const N: usize>(
    idx: usize,
    names: &'static [&'static str; N],
    objects: &[&Bound<PyAny>; N],
    cast: bool,
) -> Exception {
    let first_name = names.first().expect("Empty names slice");
    let fist_obj = objects.first().expect("Empty objects slice");

    // If the very first argument downcast
    if idx == 0 {
        return unknown_type_exception(first_name, fist_obj);
    }

    let maybe_first_f32_array = try_downcast_to_f32_array(objects[0]);
    let maybe_first_f64_array = try_downcast_to_f64_array(objects[0], cast);
    let (first_arr, first_dtype_name) = if let Some(f32_array) = maybe_first_f32_array.as_ref() {
        (f32_array.as_untyped(), f32::dtype_name())
    } else if let Some(f64_array) = maybe_first_f64_array.as_ref() {
        (f64_array.as_untyped(), f64::dtype_name())
    } else {
        return unknown_type_exception(first_name, fist_obj);
    };

    let fail_name = names.get(idx).expect("idx is out of bounds of names slice");
    let fail_obj = objects
        .get(idx)
        .expect("idx is out of bounds of objects slice");

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

pub(crate) enum GenericPyReadonlyArrays<'py, const N: usize> {
    F32([PyReadonlyArray1<'py, f32>; N]),
    F64([PyReadonlyArray1<'py, f64>; N]),
}

impl<const N: usize> GenericPyReadonlyArrays<'_, N> {
    fn array_len(&self, i: usize) -> usize {
        match self {
            Self::F32(v) => v[i].len(),
            Self::F64(v) => v[i].len(),
        }
    }
}

fn try_downcast_objects_to_f32_arrays<'py, const N: usize>(
    objects: &[&Bound<'py, PyAny>; N],
) -> [Option<PyReadonlyArray1<'py, f32>>; N] {
    let mut arrays = [const { None }; N];
    for (&obj, arr) in objects.iter().zip(arrays.iter_mut()) {
        *arr = try_downcast_to_f32_array(obj);
        // If we cannot cast an array, we stop trying for future arguments
        if arr.is_none() {
            break;
        }
    }
    arrays
}

fn try_downcast_to_f32_array<'py>(obj: &Bound<'py, PyAny>) -> Option<PyReadonlyArray1<'py, f32>> {
    let py_array = obj.downcast::<PyArray1<f32>>().ok()?;
    Some(py_array.readonly())
}

fn try_downcast_to_f64_array<'py>(
    obj: &Bound<'py, PyAny>,
    cast: bool,
) -> Option<PyReadonlyArray1<'py, f64>> {
    match (obj.downcast::<PyArray1<f64>>(), cast) {
        (Ok(py_array), _) => Some(py_array.readonly()),
        (Err(_), true) => match PyArrayLike1::<f64, AllowTypeChange>::extract(obj.as_borrowed()) {
            Ok(py_array) => Some(py_array.readonly()),
            Err(_) => None,
        },
        (Err(_), false) => None,
    }
}

const fn index<const N: usize>() -> [usize; N] {
    let mut arr = [0; N];
    let mut i = 0;
    while i < N {
        arr[i] = i;
        i += 1;
    }
    arr
}

fn downcast_objects_cast<'py, const N: usize>(
    names: &'static [&'static str; N],
    objects: &[&Bound<'py, PyAny>; N],
) -> Res<GenericPyReadonlyArrays<'py, N>> {
    let f32_arrays = try_downcast_objects_to_f32_arrays(objects);

    if f32_arrays.iter().all(|arr| arr.is_some()) {
        Ok(GenericPyReadonlyArrays::F32(
            f32_arrays.map(|arr| arr.unwrap()),
        ))
    } else {
        let result = index::<N>().map_result::<_, Exception>(|i| {
            let f64_arr = if let Some(f32_arr) = &f32_arrays[i] {
                f32_arr.cast_array::<f64>(false)?.readonly()
            } else {
                try_downcast_to_f64_array(objects[i], true)
                    .ok_or_else(|| cast_fail_reason(i, names, objects, true))?
            };
            Ok(f64_arr)
        })?;

        Ok(GenericPyReadonlyArrays::F64(result))
    }
}

fn downcast_objects_no_cast<'py, const N: usize>(
    names: &'static [&'static str; N],
    objects: &[&Bound<'py, PyAny>; N],
) -> Res<GenericPyReadonlyArrays<'py, N>> {
    let f32_arrays = try_downcast_objects_to_f32_arrays(objects);

    if f32_arrays.iter().all(|arr| arr.is_some()) {
        Ok(GenericPyReadonlyArrays::F32(
            f32_arrays.map(|arr| arr.unwrap()),
        ))
    } else {
        let mut valid_f64_count = 0;
        let f64_arrays = objects
            .map_option(|obj| {
                valid_f64_count += 1;
                try_downcast_to_f64_array(obj, false)
            })
            .ok_or_else(|| {
                let valid_f32_count = f32_arrays.iter().filter(|arr| arr.is_some()).count();
                let max_count = usize::max(valid_f32_count, valid_f64_count);
                if max_count == 0 {
                    unknown_type_exception(names[0], objects[0])
                } else {
                    let idx = max_count - 1;
                    cast_fail_reason(idx, names, objects, false)
                }
            })?;
        Ok(GenericPyReadonlyArrays::F64(f64_arrays))
    }
}

pub(crate) fn downcast_and_validate<'py, const N: usize>(
    names: &'static [&'static str; N],
    objects: &[&Bound<'py, PyAny>; N],
    check_size: &[bool; N],
    cast: bool,
) -> Res<GenericPyReadonlyArrays<'py, N>> {
    assert_eq!(names.len(), objects.len());

    let arrays = if cast {
        downcast_objects_cast(names, objects)?
    } else {
        downcast_objects_no_cast(names, objects)?
    };
    let mut first_array_len = None;
    // We checked that 1) names size matches objects size, 2) objects is not empty
    let first_name = names[0];

    for i in 1..N {
        if check_size[i] {
            let length0 = if let Some(length0) = first_array_len {
                length0
            } else {
                let length0 = arrays.array_len(0);
                first_array_len = Some(length0);
                length0
            };
            let length = arrays.array_len(i);
            if length != length0 {
                return Err(Exception::ValueError(format!(
                    "Mismatched lengths: '{}': {}, '{}': {}",
                    first_name, length0, names[i], length,
                )));
            }
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
    (@call $func:ident, $arrays:ident, $_arg1:expr,) => {{
        let [x1] = $arrays;
        $func(x1)
    }};
    (@call $func:ident, $arrays:ident, $_arg1:expr, $_arg2:expr,) => {{
        let [x1, x2] = $arrays;
        $func(x1, x2)
    }};
    (@call $func:ident, $arrays:ident, $_arg1:expr, $_arg2:expr, $_arg3:expr,) => {{
        let [x1, x2, x3] = $arrays;
        $func(x1, x2, x3)
    }};
    ($func:tt ($first_arg:expr $(,$eq:tt $arg:expr)* $(,)?)) => {
        dtype_dispatch!($func, $func, $first_arg $(,$eq $arg)*)
    };
    ($func:tt ($first_arg:expr $(,$eq:tt $arg:expr)*; cast=$cast:expr $(,)?)) => {
        dtype_dispatch!($func, $func, $first_arg $(,$eq $arg)*; cast=$cast)
    };
    ($f32:expr, $f64:expr, $first_arg:expr $(,$eq:tt $arg:expr)* $(,)?) => {
        dtype_dispatch!($f32, $f64, $first_arg $(,$eq $arg)*; cast=false)
    };
    ($f32:expr, $f64:expr, $first_arg:expr $(,$eq:tt $arg:expr)*; cast=$cast:expr) => {{
        let names = &[stringify!($first_arg), $(stringify!($arg), )*];
        let objects = &[&$first_arg, $(&$arg, )*];
        let check_size = &[ false, $(_distinguish_eq_symbol!($eq), )* ];

        let generic_arrays = crate::np_array::downcast_and_validate(names, objects, check_size, $cast)?;
        match generic_arrays {
            crate::np_array::GenericPyReadonlyArrays::F32(arrays) => {
                let func = $f32;
                dtype_dispatch!(@call func, arrays, $first_arg, $($arg,)*)
            }
            crate::np_array::GenericPyReadonlyArrays::F64(arrays) => {
                let func = $f64;
                dtype_dispatch!(@call func, arrays, $first_arg, $($arg,)*)
            }
        }
    }};
}
