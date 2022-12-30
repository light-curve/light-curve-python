use numpy::{PyReadonlyArray1, PyReadwriteArray1, PyReadwriteArray2};
use pyo3::prelude::*;
use std::convert::TryFrom;

pub(crate) type Arr<'py, T> = PyReadonlyArray1<'py, T>;

#[derive(FromPyObject)]
pub(crate) enum GenericFloatArray1<'py> {
    #[pyo3(transparent, annotation = "np.ndarray[float32]")]
    Float32(Arr<'py, f32>),
    #[pyo3(transparent, annotation = "np.ndarray[float64]")]
    Float64(Arr<'py, f64>),
}

impl<'py> TryFrom<GenericFloatArray1<'py>> for Arr<'py, f32> {
    type Error = ();

    fn try_from(value: GenericFloatArray1<'py>) -> Result<Self, Self::Error> {
        match value {
            GenericFloatArray1::Float32(a) => Ok(a),
            GenericFloatArray1::Float64(_) => Err(()),
        }
    }
}

impl<'py> TryFrom<GenericFloatArray1<'py>> for Arr<'py, f64> {
    type Error = ();

    fn try_from(value: GenericFloatArray1<'py>) -> Result<Self, Self::Error> {
        match value {
            GenericFloatArray1::Float32(_) => Err(()),
            GenericFloatArray1::Float64(a) => Ok(a),
        }
    }
}

#[derive(FromPyObject)]
pub(crate) enum GenericFloatRwArray1<'py> {
    #[pyo3(transparent, annotation = "np.ndarray[float32]")]
    Float32(PyReadwriteArray1<'py, f32>),
    #[pyo3(transparent, annotation = "np.ndarray[float64]")]
    Float64(PyReadwriteArray1<'py, f64>),
}

impl<'py> TryFrom<GenericFloatRwArray1<'py>> for PyReadwriteArray1<'py, f32> {
    type Error = ();

    fn try_from(value: GenericFloatRwArray1<'py>) -> Result<Self, Self::Error> {
        match value {
            GenericFloatRwArray1::Float32(a) => Ok(a),
            GenericFloatRwArray1::Float64(_) => Err(()),
        }
    }
}

impl<'py> TryFrom<GenericFloatRwArray1<'py>> for PyReadwriteArray1<'py, f64> {
    type Error = ();

    fn try_from(value: GenericFloatRwArray1<'py>) -> Result<Self, Self::Error> {
        match value {
            GenericFloatRwArray1::Float32(_) => Err(()),
            GenericFloatRwArray1::Float64(a) => Ok(a),
        }
    }
}

#[derive(FromPyObject)]
pub(crate) enum GenericFloatRwArray2<'py> {
    #[pyo3(transparent, annotation = "np.ndarray[float32]")]
    Float32(PyReadwriteArray2<'py, f32>),
    #[pyo3(transparent, annotation = "np.ndarray[float64]")]
    Float64(PyReadwriteArray2<'py, f64>),
}

impl<'py> TryFrom<GenericFloatRwArray2<'py>> for PyReadwriteArray2<'py, f32> {
    type Error = ();

    fn try_from(value: GenericFloatRwArray2<'py>) -> Result<Self, Self::Error> {
        match value {
            GenericFloatRwArray2::Float32(a) => Ok(a),
            GenericFloatRwArray2::Float64(_) => Err(()),
        }
    }
}

impl<'py> TryFrom<GenericFloatRwArray2<'py>> for PyReadwriteArray2<'py, f64> {
    type Error = ();

    fn try_from(value: GenericFloatRwArray2<'py>) -> Result<Self, Self::Error> {
        match value {
            GenericFloatRwArray2::Float32(_) => Err(()),
            GenericFloatRwArray2::Float64(a) => Ok(a),
        }
    }
}
