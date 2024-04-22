use crate::errors::{Exception, Res};

use enum_iterator::Sequence;
use light_curve_feature::transformers::{
    arcsinh::ArcsinhTransformer, clipped_lg::ClippedLgTransformer, identity::IdentityTransformer,
    lg::LgTransformer, ln1p::Ln1pTransformer, sqrt::SqrtTransformer, Transformer,
};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyString};

#[derive(Clone, Copy, Sequence)]
pub(crate) enum StockTransformer {
    Arcsinh,
    ClippedLg,
    Identity,
    Lg,
    Ln1p,
    Sqrt,
}

impl StockTransformer {
    pub(crate) fn all_variants() -> impl Iterator<Item = Self> {
        enum_iterator::all::<Self>()
    }

    pub(crate) fn all_names() -> impl Iterator<Item = &'static str> {
        Self::all_variants().map(|variant| variant.into())
    }

    pub(crate) fn doc(&self) -> &'static str {
        match self {
            Self::Arcsinh => ArcsinhTransformer::doc(),
            Self::ClippedLg => ClippedLgTransformer::<f64>::doc(),
            Self::Identity => IdentityTransformer::doc(),
            Self::Lg => LgTransformer::doc(),
            Self::Ln1p => Ln1pTransformer::doc(),
            Self::Sqrt => SqrtTransformer::doc(),
        }
    }
}

impl TryFrom<&str> for StockTransformer {
    type Error = Exception;

    fn try_from(s: &str) -> Res<Self> {
        Ok(match s {
            "arcsinh" => Self::Arcsinh,
            "clipped_lg" => Self::ClippedLg,
            "identity" => Self::Identity,
            "lg" => Self::Lg,
            "ln1p" => Self::Ln1p,
            "sqrt" => Self::Sqrt,
            _ => {
                return Err(Exception::ValueError(format!(
                    "Unknown stock transformer: {}",
                    s
                )))
            }
        })
    }
}

impl From<StockTransformer> for &'static str {
    fn from(val: StockTransformer) -> Self {
        match val {
            StockTransformer::Arcsinh => "arcsinh",
            StockTransformer::ClippedLg => "clipped_lg",
            StockTransformer::Identity => "identity",
            StockTransformer::Lg => "lg",
            StockTransformer::Ln1p => "ln1p",
            StockTransformer::Sqrt => "sqrt",
        }
    }
}

impl From<StockTransformer> for (Transformer<f32>, Transformer<f64>) {
    fn from(val: StockTransformer) -> Self {
        match val {
            StockTransformer::Arcsinh => (
                ArcsinhTransformer::default().into(),
                ArcsinhTransformer::default().into(),
            ),
            StockTransformer::ClippedLg => (
                ClippedLgTransformer::default().into(),
                ClippedLgTransformer::default().into(),
            ),
            StockTransformer::Identity => (
                IdentityTransformer::default().into(),
                IdentityTransformer::default().into(),
            ),
            StockTransformer::Lg => (
                LgTransformer::default().into(),
                LgTransformer::default().into(),
            ),
            StockTransformer::Ln1p => (
                Ln1pTransformer::default().into(),
                Ln1pTransformer::default().into(),
            ),
            StockTransformer::Sqrt => (
                SqrtTransformer::default().into(),
                SqrtTransformer::default().into(),
            ),
        }
    }
}

pub(crate) fn parse_transform(
    option: Option<Bound<PyAny>>,
    default: StockTransformer,
) -> Res<Option<StockTransformer>> {
    match option {
        None => Ok(None),
        Some(py_any) => {
            if let Ok(py_bool) = py_any.downcast::<PyBool>() {
                if py_bool.is_true() {
                    Ok(Some(default))
                } else {
                    Ok(None)
                }
            } else if let Ok(py_str) = py_any.downcast::<PyString>() {
                // py_str.to_str() is Python 3.10+ only
                let cow_string = py_str.to_cow()?;
                let s = cow_string.as_ref();
                if let Ok(stock_transformer) = s.try_into() {
                    Ok(Some(stock_transformer))
                } else if s == "default" {
                    Ok(Some(default))
                } else {
                    Err(Exception::ValueError(format!(
                        "Unknown transformation: {}",
                        s
                    )))
                }
            } else {
                Err(Exception::ValueError(format!(
                    "transform must be None, a bool or a str, not {}",
                    py_any.get_type().qualname()?
                )))
            }
        }
    }
}
