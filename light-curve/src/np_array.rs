use numpy::PyReadonlyArray1;
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
