use crate::errors::{Exception, Res};

use itertools::Itertools;
use light_curve_feature::Float;
use ndarray::{ArrayView1, Zip};

pub fn is_sorted<T>(a: &[T]) -> bool
where
    T: PartialOrd,
{
    a.iter().tuple_windows().all(|(a, b)| a < b)
}

pub fn check_sorted<T>(a: &[T], sorted: Option<bool>) -> Res<()>
where
    T: PartialOrd,
{
    match sorted {
        Some(true) => Ok(()),
        Some(false) => Err(Exception::NotImplementedError(String::from(
            "sorting is not implemented, please provide time-sorted arrays",
        ))),
        None => {
            if is_sorted(a) {
                Ok(())
            } else {
                Err(Exception::ValueError(String::from(
                    "t must be in ascending order",
                )))
            }
        }
    }
}

pub fn check_finite<T>(a: ArrayView1<'_, T>) -> Res<()>
where
    T: Float,
{
    if Zip::from(a).all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(Exception::ValueError(String::from(
            "t and m values must be finite",
        )))
    }
}

pub fn check_no_nans<T>(a: ArrayView1<'_, T>) -> Res<()>
where
    T: Float,
{
    // There are no Zip::any
    if Zip::from(a).all(|x| !x.is_nan()) {
        Ok(())
    } else {
        Err(Exception::ValueError(String::from(
            "input arrays must not contain any NaNs",
        )))
    }
}
