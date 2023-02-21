use pyo3::exceptions::{
    PyIndexError, PyNotImplementedError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::import_exception;
use pyo3::PyErr;
use std::fmt::Debug;
use std::result::Result;
use thiserror::Error;

import_exception!(pickle, PicklingError);
import_exception!(pickle, UnpicklingError);

#[allow(clippy::enum_variant_names)]
#[derive(Clone, Error, Debug)]
#[error("{0}")]
pub(crate) enum Exception {
    // builtins
    IndexError(String),
    NotImplementedError(String),
    RuntimeError(String),
    TypeError(String),
    ValueError(String),
    // pickle
    PicklingError(String),
    UnpicklingError(String),
}

impl From<Exception> for PyErr {
    fn from(err: Exception) -> PyErr {
        match err {
            // builtins
            Exception::IndexError(err) => PyIndexError::new_err(err),
            Exception::NotImplementedError(err) => PyNotImplementedError::new_err(err),
            Exception::RuntimeError(err) => PyRuntimeError::new_err(err),
            Exception::TypeError(err) => PyTypeError::new_err(err),
            Exception::ValueError(err) => PyValueError::new_err(err),
            // pickle
            Exception::PicklingError(err) => PicklingError::new_err(err),
            Exception::UnpicklingError(err) => UnpicklingError::new_err(err),
        }
    }
}

pub(crate) type Res<T> = Result<T, Exception>;
