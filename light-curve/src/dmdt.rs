use crate::check::check_sorted;
use crate::cont_array::{ContArray, ContCowArray};
use crate::errors::{Exception, Res};
use crate::np_array::{Arr, GenericFloatArray1};

use conv::{ApproxFrom, ApproxInto, ConvAsUtil};
use enumflags2::{bitflags, BitFlags};
use light_curve_dmdt as lcdmdt;
use ndarray::IntoNdProducer;
use numpy::{Element, IntoPyArray, PyArray1, ToPyArray};
use pyo3::class::iter::PyIterProtocol;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::cell::RefCell;
use std::convert::{TryFrom, TryInto};
use std::ops::{DerefMut, Range};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use unzip3::Unzip3;

#[derive(FromPyObject, Copy, Clone, std::fmt::Debug)]
enum DropNObsType {
    #[pyo3(transparent, annotation = "int")]
    Int(usize),
    #[pyo3(transparent, annotation = "float")]
    Float(f64),
}

#[bitflags]
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq)]
enum NormFlag {
    Dt,
    Max,
}

#[derive(Copy, Clone, Debug)]
enum ErrorFunction {
    Exact,
    Eps1Over1e3,
}

#[derive(Clone)]
struct GenericDmDt<T>
where
    T: lcdmdt::Float,
{
    dmdt: lcdmdt::DmDt<T>,
    norm: BitFlags<NormFlag>,
    error_func: ErrorFunction,
    n_jobs: usize,
}

impl<'a, T> GenericDmDt<T>
where
    T: Element + ndarray::NdFloat + lcdmdt::ErfFloat,
    Arr<'a, T>: TryFrom<GenericFloatArray1<'a>>,
{
    fn sigma_to_err2(sigma: Arr<'a, T>) -> ContArray<T> {
        let mut a: ContArray<_> = sigma.as_array().into();
        a.0.mapv_inplace(|x| x.powi(2));
        a
    }

    fn normalize(&self, a: &mut ndarray::Array2<T>, t: &[T]) {
        if self.norm.contains(NormFlag::Dt) {
            let dt = self.dmdt.dt_points(t);
            let dt_no_zeros = dt.mapv(|x| {
                if x == 0 {
                    T::one()
                } else {
                    x.approx_into().unwrap()
                }
            });
            *a /= &dt_no_zeros.into_shape((a.nrows(), 1)).unwrap();
        }
        if self.norm.contains(NormFlag::Max) {
            let max = *a.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            if !max.is_zero() {
                a.mapv_inplace(|x| x / max);
            }
        }
    }

    fn py_count_dt(&self, py: Python, t: Arr<'a, T>, sorted: Option<bool>) -> Res<PyObject> {
        self.count_dt(
            ContCowArray::from_view(t.as_array(), true).as_slice(),
            sorted,
        )
        .map(|a| a.into_pyarray(py).into_py(py))
    }

    fn count_dt(&self, t: &[T], sorted: Option<bool>) -> Res<ndarray::Array1<T>> {
        check_sorted(t, sorted)?;
        Ok(self.dmdt.dt_points(t).mapv(|x| x.approx_into().unwrap()))
    }

    fn py_count_dt_many(
        &self,
        py: Python,
        t_: Vec<GenericFloatArray1<'a>>,
        sorted: Option<bool>,
    ) -> Res<PyObject> {
        let wrapped_t_ = t_
            .into_iter()
            .enumerate()
            .map(|(i, t)| {
                let t: Result<Arr<_>, _> = t.try_into();
                match t {
                    Ok(t) => Ok(t),
                    Err(_) => Err(Exception::TypeError(format!(
                        "t_[{}] has mismatched dtype with the t_[0] which is {}",
                        i,
                        std::any::type_name::<T>()
                    ))),
                }
            })
            .collect::<Res<Vec<_>>>()?;
        let array_t_ = wrapped_t_
            .iter()
            .map(|t| ContCowArray::from_view(t.as_array(), true))
            .collect::<Vec<_>>();
        let typed_t_ = array_t_.iter().map(|t| t.as_slice()).collect();
        Ok(self
            .count_dt_many(typed_t_, sorted)?
            .into_pyarray(py)
            .into_py(py))
    }

    fn count_dt_many(&self, t_: Vec<&[T]>, sorted: Option<bool>) -> Res<ndarray::Array2<T>> {
        let dt_size = self.dmdt.dt_grid.cell_count();
        let mut result = ndarray::Array2::zeros((t_.len(), dt_size));

        rayon::ThreadPoolBuilder::new()
            .num_threads(self.n_jobs)
            .build()
            .unwrap()
            .install(|| {
                ndarray::Zip::from(result.outer_iter_mut())
                    .and(t_.into_producer())
                    .into_par_iter()
                    .try_for_each::<_, Res<_>>(|(mut count, t)| {
                        count.assign(&self.count_dt(t, sorted)?);
                        Ok(())
                    })
            })?;
        Ok(result)
    }

    fn py_points(
        &self,
        py: Python,
        t: Arr<'a, T>,
        m: Arr<'a, T>,
        sorted: Option<bool>,
    ) -> Res<PyObject> {
        Ok(self
            .points(
                ContCowArray::from_view(t.as_array(), true).as_slice(),
                ContCowArray::from_view(m.as_array(), true).as_slice(),
                sorted,
            )?
            .into_pyarray(py)
            .into_py(py))
    }

    fn points(&self, t: &[T], m: &[T], sorted: Option<bool>) -> Res<ndarray::Array2<T>> {
        check_sorted(t, sorted)?;

        let mut result = self.dmdt.points(t, m).mapv(|x| x.approx_into().unwrap());
        self.normalize(&mut result, t);
        Ok(result)
    }

    fn py_points_many(
        &self,
        py: Python,
        lcs: Vec<(GenericFloatArray1<'a>, GenericFloatArray1<'a>)>,
        sorted: Option<bool>,
    ) -> Res<PyObject> {
        let wrapped_lcs = lcs
            .into_iter()
            .enumerate()
            .map(|(i, (t, m))| {
                let t: Result<Arr<'a, T>, _> = t.try_into();
                let m: Result<Arr<'a, T>, _> = m.try_into();
                match (t, m) {
                    (Ok(t), Ok(m)) => Ok((t, m)),
                    _ => Err(Exception::TypeError(format!(
                        "lcs[{}] elements have mismatched dtype with the lc[0][0] which is {}",
                        i,
                        std::any::type_name::<T>(),
                    ))),
                }
            })
            .collect::<Res<Vec<_>>>()?;
        let array_lcs = wrapped_lcs
            .iter()
            .map(|(t, m)| {
                (
                    ContCowArray::from_view(t.as_array(), true),
                    ContCowArray::from_view(m.as_array(), true),
                )
            })
            .collect::<Vec<_>>();
        let typed_lcs = array_lcs
            .iter()
            .map(|(t, m)| (t.as_slice(), m.as_slice()))
            .collect();
        Ok(self
            .points_many(typed_lcs, sorted)?
            .into_pyarray(py)
            .into_py(py))
    }

    fn points_many(&self, lcs: Vec<(&[T], &[T])>, sorted: Option<bool>) -> Res<ndarray::Array3<T>> {
        let dmdt_shape = self.dmdt.shape();
        let mut result = ndarray::Array3::zeros((lcs.len(), dmdt_shape.0, dmdt_shape.1));

        rayon::ThreadPoolBuilder::new()
            .num_threads(self.n_jobs)
            .build()
            .unwrap()
            .install(|| {
                ndarray::Zip::from(result.outer_iter_mut())
                    .and(lcs.into_producer())
                    .into_par_iter()
                    .try_for_each::<_, Res<_>>(|(mut map, (t, m))| {
                        map.assign(&self.points(t, m, sorted)?);
                        Ok(())
                    })
            })?;
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn generic_dmdt_points_batches(
        &self,
        lcs: Vec<(GenericFloatArray1<'a>, GenericFloatArray1<'a>)>,
        sorted: Option<bool>,
        batch_size: usize,
        yield_index: bool,
        shuffle: bool,
        drop_nobs: DropNObsType,
        random_seed: Option<u64>,
    ) -> Res<GenericDmDtBatches<T, TmLc<T>>> {
        let typed_lcs = lcs
            .into_iter()
            .enumerate()
            .map(|(i, (t, m))| {
                let t: Result<Arr<'a, T>, _> = t.try_into();
                let m: Result<Arr<'a, T>, _> = m.try_into();
                match (t, m) {
                    (Ok(t), Ok(m)) => {
                        let t: ContArray<_> = t.as_array().into();
                        check_sorted(t.as_slice(), sorted)?;
                        let m: ContArray<_> = m.as_array().into();
                        Ok((t, m))
                    }
                    _ => Err(Exception::TypeError(format!(
                        "lcs[{}] elements have mismatched dtype with the lc[0][0] which is {}",
                        i,
                        std::any::type_name::<T>(),
                    ))),
                }
            })
            .collect::<Res<Vec<_>>>()?;
        GenericDmDtBatches::new(
            self.clone(),
            typed_lcs,
            batch_size,
            yield_index,
            shuffle,
            drop_nobs,
            random_seed,
        )
    }

    fn py_gausses(
        &self,
        py: Python,
        t: Arr<'a, T>,
        m: Arr<'a, T>,
        sigma: Arr<'a, T>,
        sorted: Option<bool>,
    ) -> Res<PyObject> {
        let err2 = Self::sigma_to_err2(sigma);
        Ok(self
            .gausses(
                ContCowArray::from_view(t.as_array(), true).as_slice(),
                ContCowArray::from_view(m.as_array(), true).as_slice(),
                err2.as_slice(),
                sorted,
            )?
            .into_pyarray(py)
            .into_py(py))
    }

    fn gausses(
        &self,
        t: &[T],
        m: &[T],
        err2: &[T],
        sorted: Option<bool>,
    ) -> Res<ndarray::Array2<T>> {
        check_sorted(t, sorted)?;

        let mut result = match self.error_func {
            ErrorFunction::Exact => self.dmdt.gausses::<lcdmdt::ExactErf>(t, m, err2),
            ErrorFunction::Eps1Over1e3 => self.dmdt.gausses::<lcdmdt::Eps1Over1e3Erf>(t, m, err2),
        };
        self.normalize(&mut result, t);
        Ok(result)
    }

    fn py_gausses_many(
        &self,
        py: Python,
        lcs: Vec<(
            GenericFloatArray1<'a>,
            GenericFloatArray1<'a>,
            GenericFloatArray1<'a>,
        )>,
        sorted: Option<bool>,
    ) -> Res<PyObject> {
        let wrapped_lcs = lcs
            .into_iter()
            .enumerate()
            .map(|(i, (t, m, sigma))| {
                let t: Result<Arr<_>, _> = t.try_into();
                let m: Result<Arr<_>, _> = m.try_into();
                let sigma: Result<Arr<_>, _> = sigma.try_into();

                match (t, m, sigma) {
                    (Ok(t), Ok(m), Ok(sigma)) => Ok((t, m, Self::sigma_to_err2(sigma))),
                    _ => Err(Exception::TypeError(format!(
                        "lcs[{}] elements have mismatched dtype with the lc[0][0] which is {}",
                        i,
                        std::any::type_name::<T>()
                    ))),
                }
            })
            .collect::<Res<Vec<_>>>()?;
        let array_lcs = wrapped_lcs
            .iter()
            .map(|(t, m, err2)| {
                (
                    ContCowArray::from_view(t.as_array(), true),
                    ContCowArray::from_view(m.as_array(), true),
                    err2,
                )
            })
            .collect::<Vec<_>>();
        let typed_lcs = array_lcs
            .iter()
            .map(|(t, m, err2)| (t.as_slice(), m.as_slice(), err2.as_slice()))
            .collect();
        Ok(self
            .gausses_many(typed_lcs, sorted)?
            .into_pyarray(py)
            .into_py(py))
    }

    fn gausses_many(
        &self,
        lcs: Vec<(&[T], &[T], &[T])>,
        sorted: Option<bool>,
    ) -> Res<ndarray::Array3<T>> {
        let dmdt_shape = self.dmdt.shape();
        let mut result = ndarray::Array3::zeros((lcs.len(), dmdt_shape.0, dmdt_shape.1));

        rayon::ThreadPoolBuilder::new()
            .num_threads(self.n_jobs)
            .build()
            .unwrap()
            .install(|| {
                ndarray::Zip::from(result.outer_iter_mut())
                    .and(lcs.into_producer())
                    .into_par_iter()
                    .try_for_each::<_, Res<_>>(|(mut map, (t, m, err2))| {
                        map.assign(&self.gausses(t, m, err2, sorted)?);
                        Ok(())
                    })
            })?;
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn generic_dmdt_gausses_batches(
        &self,
        lcs: Vec<(
            GenericFloatArray1<'a>,
            GenericFloatArray1<'a>,
            GenericFloatArray1<'a>,
        )>,
        sorted: Option<bool>,
        batch_size: usize,
        yield_index: bool,
        shuffle: bool,
        drop_nobs: DropNObsType,
        random_seed: Option<u64>,
    ) -> Res<GenericDmDtBatches<T, Tmerr2Lc<T>>> {
        let typed_lcs = lcs
            .into_iter()
            .enumerate()
            .map(|(i, (t, m, sigma))| {
                let t: Result<Arr<'a, T>, _> = t.try_into();
                let m: Result<Arr<'a, T>, _> = m.try_into();
                let sigma: Result<Arr<'a, T>, _> = sigma.try_into();
                match (t, m, sigma) {
                    (Ok(t), Ok(m), Ok(sigma)) => {
                        let t: ContArray<_> = t.as_array().into();
                        check_sorted(t.as_slice(), sorted)?;
                        let m: ContArray<_> = m.as_array().into();
                        let err2 = Self::sigma_to_err2(sigma);
                        Ok((t, m, err2))
                    }
                    _ => Err(Exception::TypeError(format!(
                        "lcs[{}] elements have mismatched dtype with the lc[0][0] which is {}",
                        i,
                        std::any::type_name::<T>(),
                    ))),
                }
            })
            .collect::<Res<Vec<_>>>()?;
        GenericDmDtBatches::new(
            self.clone(),
            typed_lcs,
            batch_size,
            yield_index,
            shuffle,
            drop_nobs,
            random_seed,
        )
    }
}

struct GenericDmDtBatches<T, LC>
where
    T: lcdmdt::Float,
{
    dmdt: GenericDmDt<T>,
    lcs: Vec<LC>,
    batch_size: usize,
    yield_index: bool,
    shuffle: bool,
    drop_nobs: Option<DropNObsType>,
    rng: Mutex<Xoshiro256PlusPlus>,
}

impl<T, LC> GenericDmDtBatches<T, LC>
where
    T: lcdmdt::Float,
{
    fn new(
        dmdt: GenericDmDt<T>,
        lcs: Vec<LC>,
        batch_size: usize,
        yield_index: bool,
        shuffle: bool,
        drop_nobs: DropNObsType,
        random_seed: Option<u64>,
    ) -> Res<Self> {
        let rng = match random_seed {
            Some(seed) => Xoshiro256PlusPlus::seed_from_u64(seed),
            None => Xoshiro256PlusPlus::from_rng(&mut rand::thread_rng()).unwrap(),
        };
        let drop_nobs = match drop_nobs {
            DropNObsType::Int(0) => None,
            DropNObsType::Int(_) => Some(drop_nobs),
            DropNObsType::Float(x) => match x {
                _ if x == 0.0 => None,
                _ if (0.0..1.0).contains(&x) => Some(drop_nobs),
                _ => {
                    return Err(Exception::ValueError(String::from(
                        "if drop_nobs is float, it must be in [0.0, 1.0)",
                    )))
                }
            },
        };
        Ok(Self {
            dmdt,
            lcs,
            batch_size,
            yield_index,
            shuffle,
            drop_nobs,
            rng: Mutex::new(rng),
        })
    }

    fn dropped_index<R: rand::Rng>(&self, rng: &mut R, length: usize) -> Res<Vec<usize>> {
        let drop_nobs = match self.drop_nobs {
            Some(drop_nobs) => drop_nobs,
            None => {
                return Err(Exception::RuntimeError(String::from(
                    "dropping is not required: drop_nobs = 0",
                )))
            }
        };
        let drop_nobs = match drop_nobs {
            DropNObsType::Int(x) => x,
            DropNObsType::Float(x) => f64::round(x * length as f64) as usize,
        };
        if drop_nobs >= length {
            return Err(Exception::ValueError(format!(
                "cannot drop {} observations from light curve containing {} points",
                drop_nobs, length
            )));
        }
        if drop_nobs == 0 {
            return Ok((0..length).collect());
        }
        let mut idx = rand::seq::index::sample(rng, length, length - drop_nobs).into_vec();
        idx.sort_unstable();
        Ok(idx)
    }
}

macro_rules! dmdt_batches {
    ($worker: expr, $generic: ty, $name_batches: ident, $name_iter: ident, $t: ty $(,)?) => {
        #[pyclass(module = "light_curve.light_curve_ext")]
        struct $name_batches {
            dmdt_batches: Arc<$generic>,
        }

        #[pyproto]
        impl PyIterProtocol for $name_batches {
            fn __iter__(slf: PyRef<Self>) -> PyResult<Py<$name_iter>> {
                let iter = $name_iter::new(slf.dmdt_batches.clone());
                Py::new(slf.py(), iter)
            }
        }

        #[pyclass(module = "light_curve.light_curve_ext")]
        struct $name_iter {
            dmdt_batches: Arc<$generic>,
            lcs_order: Vec<usize>,
            range: Range<usize>,
            worker_thread: RefCell<Option<JoinHandle<Res<ndarray::Array3<$t>>>>>,
            rng: Option<Xoshiro256PlusPlus>,
        }

        impl $name_iter {
            fn child_rng(rng: Option<&mut Xoshiro256PlusPlus>) -> Option<Xoshiro256PlusPlus> {
                rng.map(|rng| Xoshiro256PlusPlus::from_rng(rng).unwrap())
            }

            fn new(dmdt_batches: Arc<$generic>) -> Self {
                let range = 0..usize::min(dmdt_batches.batch_size, dmdt_batches.lcs.len());

                let lcs_order = match dmdt_batches.shuffle {
                    false => (0..dmdt_batches.lcs.len()).collect(),
                    true => rand::seq::index::sample(
                        dmdt_batches.rng.lock().unwrap().deref_mut(),
                        dmdt_batches.lcs.len(),
                        dmdt_batches.lcs.len(),
                    )
                    .into_vec(),
                };

                let mut rng = match dmdt_batches.drop_nobs {
                    Some(_) => {
                        let mut parent_rng = dmdt_batches.rng.lock().unwrap();
                        Self::child_rng(Some(parent_rng.deref_mut()))
                    }
                    None => None,
                };

                let worker_thread = RefCell::new(Some(Self::run_worker_thread(
                    &dmdt_batches,
                    &lcs_order[range.start..range.end],
                    Self::child_rng(rng.as_mut()),
                )));

                Self {
                    dmdt_batches,
                    lcs_order,
                    range,
                    worker_thread,
                    rng,
                }
            }

            fn run_worker_thread(
                dmdt_batches: &Arc<$generic>,
                indexes: &[usize],
                rng: Option<Xoshiro256PlusPlus>,
            ) -> JoinHandle<Res<ndarray::Array3<$t>>> {
                let dmdt_batches = dmdt_batches.clone();
                let indexes = indexes.to_vec();
                std::thread::spawn(move || Self::worker(dmdt_batches, &indexes, rng))
            }

            fn worker(
                dmdt_batches: Arc<$generic>,
                indexes: &[usize],
                rng: Option<Xoshiro256PlusPlus>,
            ) -> Res<ndarray::Array3<$t>> {
                $worker(dmdt_batches, indexes, rng)
            }
        }

        impl Drop for $name_iter {
            fn drop(&mut self) {
                let t = self.worker_thread.replace(None);
                t.into_iter().for_each(|t| drop(t.join().unwrap()));
            }
        }

        #[pyproto]
        impl PyIterProtocol for $name_iter {
            fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
                slf
            }

            fn __next__(mut slf: PyRefMut<Self>) -> Res<Option<PyObject>> {
                if slf.worker_thread.borrow().is_none() {
                    return Ok(None);
                }

                let current_range = match slf.dmdt_batches.yield_index {
                    true => Some(slf.range.clone()),
                    false => None,
                };

                slf.range = slf.range.end
                    ..usize::min(
                        slf.range.end + slf.dmdt_batches.batch_size,
                        slf.dmdt_batches.lcs.len(),
                    );

                // Replace with None temporary to not have two workers running simultaneously
                let array = slf.worker_thread.replace(None).unwrap().join().unwrap()?;
                if !slf.range.is_empty() {
                    let rng = Self::child_rng(slf.rng.as_mut());
                    slf.worker_thread.replace(Some(Self::run_worker_thread(
                        &slf.dmdt_batches,
                        &slf.lcs_order[slf.range.start..slf.range.end],
                        rng,
                    )));
                }

                let py_array = array.into_pyarray(slf.py()).into_py(slf.py());
                match current_range {
                    Some(range) => {
                        let py_index =
                            PyArray1::from_slice(slf.py(), &slf.lcs_order[range.start..range.end])
                                .into_py(slf.py());
                        let tuple = PyTuple::new(slf.py(), &[py_index, py_array]).into_py(slf.py());
                        Ok(Some(tuple))
                    }
                    None => Ok(Some(py_array)),
                }
            }
        }
    };
}

macro_rules! py_dmdt_batches {
    ($worker: expr, $($generic: ty, $name_batches: ident, $name_iter: ident, $t: ty),* $(,)?) => {
        $(
            dmdt_batches!($worker, $generic, $name_batches, $name_iter, $t);
        )*
    };
}

type TmLc<T> = (ContArray<T>, ContArray<T>);
type Tmerr2Lc<T> = (ContArray<T>, ContArray<T>, ContArray<T>);

py_dmdt_batches!(
    |dmdt_batches: Arc<GenericDmDtBatches<_, (ContArray<_>, ContArray<_>)>>, indexes: &[usize], rng: Option<Xoshiro256PlusPlus>| {
        let mut lcs: Vec<_> = indexes
            .iter()
            .map(|&i| {
                let (t, m) = &dmdt_batches.lcs[i];
                (
                    t.as_slice(),
                    m.as_slice(),
                )
            })
            .collect();
        let dropped_owned_lcs = match (dmdt_batches.drop_nobs, rng) {
            (Some(_), Some(mut rng)) => {
                let owned_lcs: Vec<(Vec<_>, Vec<_>)> = lcs.iter().map(|(t, m)| {
                    Ok(dmdt_batches.dropped_index(&mut rng, t.len())?.iter().map(|&i| (t[i], m[i])).unzip())
                }).collect::<Res<_>>()?;
                Some(owned_lcs)
            },
            (None, None) => None,
            (_, _) => return Err(Exception::RuntimeError(String::from("cannot be here, please report an issue")))
        };
        if let Some(owned_lcs) = &dropped_owned_lcs {
            for (ref_lc, lc) in lcs.iter_mut().zip(owned_lcs) {
                *ref_lc = (lc.0.as_slice(), lc.1.as_slice());
            }
        }
        dmdt_batches.dmdt.points_many(lcs, Some(true))
    },
    GenericDmDtBatches<f32, TmLc<f32>>,
    DmDtPointsBatchesF32,
    DmDtPointsIterF32,
    f32,
    GenericDmDtBatches<f64, TmLc<f64>>,
    DmDtPointsBatchesF64,
    DmDtPointsIterF64,
    f64,
);

py_dmdt_batches!(
    |dmdt_batches: Arc<GenericDmDtBatches<_, (ContArray<_>, ContArray<_>, ContArray<_>)>>, indexes: &[usize], rng: Option<Xoshiro256PlusPlus>| {
        let mut lcs: Vec<_> = indexes
            .iter()
            .map(|&i| {
                let (t, m, err2) = &dmdt_batches.lcs[i];
                (
                    t.as_slice(),
                    m.as_slice(),
                    err2.as_slice(),
                )
            })
            .collect();
        let dropped_owned_lcs = match (dmdt_batches.drop_nobs, rng) {
            (Some(_), Some(mut rng)) => {
                let owned_lcs: Vec<(Vec<_>, Vec<_>, Vec<_>)> = lcs.iter().map(|(t, m, err2)| {
                    Ok(dmdt_batches.dropped_index(&mut rng, t.len())?.iter().map(|&i| (t[i], m[i], err2[i])).unzip3())
                }).collect::<Res<_>>()?;
                Some(owned_lcs)
            },
            (None, None) => None,
            (_, _) => return Err(Exception::RuntimeError(String::from("cannot be here, please report an issue")))
        };
        if let Some(owned_lcs) = &dropped_owned_lcs {
            for (ref_lc, lc) in lcs.iter_mut().zip(owned_lcs) {
                *ref_lc = (lc.0.as_slice(), lc.1.as_slice(), lc.2.as_slice());
            }
        }
        dmdt_batches.dmdt.gausses_many(lcs, Some(true))
    },
    GenericDmDtBatches<f32, Tmerr2Lc<f32>>,
    DmDtGaussesBatchesF32,
    DmDtGaussesIterF32,
    f32,
    GenericDmDtBatches<f64, Tmerr2Lc<f64>>,
    DmDtGaussesBatchesF64,
    DmDtGaussesIterF64,
    f64,
);

/// dm-dt map producer
///
/// Each pair of observations is mapped to dm-dt plane bringing unity value. dmdt-map is a rectangle
/// on this plane consisted of `dt_size` x `dm_size` cells, and limited by `[min_dt; max_dt)` and
/// `[min_dm; max_dm)` intervals. `.points*()` methods assigns unity value of each observation to a
/// single cell, while `.gausses*()` methods smears this unity value over all cells with given dt
/// value using normal distribution `N(m2 - m1, sigma1^2 + sigma2^2)`, where `(t1, m1, sigma1)` and
/// `(t2, m2, sigma2)` are a pair of observations including uncertainties. Optionally, after the map
/// is built, normalisation is performed ("norm" parameter): "dt" means divide each dt = const
/// column by the total number of all observations corresponded to given dt (in this case
/// `gausses()` output can be interpreted as conditional probability p(dm|dt)); "max" means divide
/// all values by the maximum value; both options can be combined, then "max" is performed after
/// "dt".
///
/// Parameters
/// ----------
/// dt : np.array of float64
///     Ascending array of dt grid edges
/// dm : np.array of float64
///     Ascending array of dm grid edges
/// dt_type : str, optional
///     Type of `dt` grid, one of:
///     - 'auto' (default) means check if grid is linear or logarithmic one,
///       which allows some speed-up
///     - 'linear' says to build a linear grid from the first and last values
///       of `dt`, using the same number of edges
///     - 'log' is the same as 'linear' but for building logarithmic grid
///     - 'asis' means using the given array as a grid
/// dm_type : str, optional
///     Type of `dm` grid, see `dt_type` for details
/// norm : list of str, optional
///     Types of normalisation, cab be any combination of "dt" and "max",
///     default is an empty list `[]` which means no normalisation
/// n_jobs : int, optional
///     Number of parallel threads to run bulk methods such as `points_many()`
///     or `gausses_batches()` default is `-1` which means to use as many
///     threads as CPU cores
/// approx_erf : bool, optional
///     Use approximation normal CDF in `gausses*` methods, reduces accuracy,
///     but has better performance, default is `False`
///
/// Attributes
/// ----------
/// n_jobs : int
/// shape : (int, int)
///     Shape of a single dmdt map, `(dt_size, dm_size)`
/// dt_grid : np.array of float64
/// min_dt : float
/// max_dt : float
/// dm_grid : np.array of float64
/// min_dm : float
/// max_dm : float
///
/// Methods
/// -------
/// from_borders(min_lgdt, max_lgdt, max_abs_dm, lgdt_size, dm_size, **kwargs)
///     Construct `DmDt` with logarithmic dt grid [10^min_lgdt, 10^max_lgdt)
///     and linear dm grid [-max_abs_dm, max_abs_dm), `kwargs` are passed to
///     `__new__()`
/// points(t, m, sorted=None)
///     Produces dmdt-maps from light curve
/// gausses(t, m, sigma, sorted=None)
///     Produces smeared dmdt-map from noisy light curve
/// count_dt(t, sorted=None)
///     Total number of observations per each dt interval
/// points_many(lcs, sorted=None)
///     Produces dmdt-maps from a list of light curves
/// gausses_many(lcs, sorted=None)
///     Produces smeared dmdt-maps from a list of light curves
/// count_dt_many(t_, sorted=None)
///     Number of observations in each dt for a list of arrays
/// points_batches(lcs, sorted=None, batch_size=1, yield_index=False, shuffle=False, drop_nobs=0, random_seed=None)
///     Gives a reusable iterable which yields dmdt-maps
/// gausses_batches(lcs, sorted=None, batch_size=1, yield_index=False, shuffle=False, drop_nobs=0, random_seed=None)
///     Gives a reusable iterable which yields smeared dmdt-maps
///
#[pyclass(module = "light_curve.light_curve_ext")]
pub struct DmDt {
    dmdt_f64: GenericDmDt<f64>,
    dmdt_f32: GenericDmDt<f32>,
}

#[derive(Clone, Copy)]
enum GridType {
    Linear,
    Log,
    Generic,
}

impl DmDt {
    // clippy has false-positive report for 1..-1 ranges inside s!
    // https://github.com/rust-lang/rust-clippy/issues/5808
    #[allow(clippy::reversed_empty_ranges)]
    fn grid_type(a: &ndarray::ArrayView1<f64>) -> Res<GridType> {
        const EPS: f64 = 1000.0 * f64::EPSILON;

        if !ndarray::Zip::from(a.slice(ndarray::s![..-1]))
            .and(a.slice(ndarray::s![1..]))
            .all(|&x, &y| x < y)
        {
            return Err(Exception::ValueError(
                "dmdt grid must be in ascending order".to_owned(),
            ));
        }

        let a1 = *a
            .get(1)
            .ok_or_else(|| Exception::IndexError("index 1 is out of bounds".to_owned()))?;
        let a0 = *a.get(0).unwrap();

        {
            let step = a1 - a0;
            if ndarray::Zip::from(a.slice(ndarray::s![1..-1]))
                .and(a.slice(ndarray::s![2..]))
                .all(|&x, &y| f64::abs((step - y + x) / step) < EPS)
            {
                return Ok(GridType::Linear);
            }
        }
        {
            let ln_step = f64::ln(a1 / a0);
            if ndarray::Zip::from(a.slice(ndarray::s![1..-1]))
                .and(a.slice(ndarray::s![2..]))
                .all(|&x, &y| f64::abs((ln_step - f64::ln(y / x)) / ln_step) < EPS)
            {
                return Ok(GridType::Log);
            }
        }
        Ok(GridType::Generic)
    }

    fn array_to_grid<T>(
        x: &ndarray::ArrayView1<f64>,
        grid_type: GridType,
    ) -> Res<Box<dyn lcdmdt::Grid<T>>>
    where
        T: lcdmdt::Float + ApproxFrom<f64>,
    {
        Ok(match grid_type {
            GridType::Linear => Box::new(lcdmdt::LinearGrid::new(
                (*x.get(0).unwrap()).approx().unwrap(),
                (*x.get(x.len() - 1).unwrap()).approx().unwrap(),
                x.len() - 1,
            )),
            GridType::Log => Box::new(lcdmdt::LgGrid::from_start_end(
                (*x.get(0).unwrap()).approx().unwrap(),
                (*x.get(x.len() - 1).unwrap()).approx().unwrap(),
                x.len() - 1,
            )),
            GridType::Generic => Box::new(
                lcdmdt::ArrayGrid::new(x.mapv(|x| x.approx().unwrap()))
                    .map_err(|err| Exception::ValueError(err.to_string()))?,
            ),
        })
    }

    fn from_dmdts(
        dmdt_f32: lcdmdt::DmDt<f32>,
        dmdt_f64: lcdmdt::DmDt<f64>,
        norm: Vec<&str>,
        n_jobs: i64,
        approx_erf: bool,
    ) -> Res<Self> {
        let norm = norm
            .iter()
            .map(|&s| match s {
                "dt" => Ok(NormFlag::Dt),
                "max" => Ok(NormFlag::Max),
                _ => Err(Exception::ValueError(format!(
                    "normalisation name {:?} is unknown, known names are: \"dt\", \"norm\"",
                    s
                ))),
            })
            .collect::<Res<BitFlags<NormFlag>>>()?;
        let error_func = match approx_erf {
            true => ErrorFunction::Eps1Over1e3,
            false => ErrorFunction::Exact,
        };
        let n_jobs = if n_jobs <= 0 {
            num_cpus::get()
        } else {
            n_jobs as usize
        };
        Ok(Self {
            dmdt_f32: GenericDmDt {
                dmdt: dmdt_f32,
                norm,
                error_func,
                n_jobs,
            },
            dmdt_f64: GenericDmDt {
                dmdt: dmdt_f64,
                norm,
                error_func,
                n_jobs,
            },
        })
    }
}

#[pymethods]
impl DmDt {
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[args(
        dt,
        dm,
        dt_type = "\"auto\"",
        dm_type = "\"auto\"",
        norm = "vec![]",
        n_jobs = -1,
        approx_erf = "false"
    )]
    fn __new__<'a>(
        dt: Arr<'a, f64>,
        dm: Arr<'a, f64>,
        dm_type: &str,
        dt_type: &str,
        norm: Vec<&str>,
        n_jobs: i64,
        approx_erf: bool,
    ) -> Res<Self> {
        let dt = dt.as_array();
        let dm = dm.as_array();

        let grid_type_dt = match dt_type {
            "auto" => Self::grid_type(&dt)?,
            "linear" => GridType::Linear,
            "log" => GridType::Log,
            "asis" => GridType::Generic,
            _ => {
                return Err(Exception::ValueError(
                    "dt_type must be 'auto', 'linear', 'log' or 'asis'".to_owned(),
                ))
            }
        };
        let grid_type_dm = match dm_type {
            "auto" => Self::grid_type(&dm)?,
            "linear" => GridType::Linear,
            "log" => GridType::Log,
            "asis" => GridType::Generic,
            _ => {
                return Err(Exception::ValueError(
                    "dm_type must be 'auto', 'linear', 'log' or 'asis'".to_owned(),
                ))
            }
        };

        let dmdt_f32: lcdmdt::DmDt<f32> = lcdmdt::DmDt {
            dt_grid: Self::array_to_grid(&dt, grid_type_dt)?,
            dm_grid: Self::array_to_grid(&dm, grid_type_dm)?,
        };
        let dmdt_f64: lcdmdt::DmDt<f64> = lcdmdt::DmDt {
            dt_grid: Self::array_to_grid(&dt, grid_type_dt)?,
            dm_grid: Self::array_to_grid(&dm, grid_type_dm)?,
        };

        Self::from_dmdts(dmdt_f32, dmdt_f64, norm, n_jobs, approx_erf)
    }

    #[allow(clippy::too_many_arguments)]
    #[staticmethod]
    #[args(
        min_lgdt,
        max_lgdt,
        max_abs_dm,
        lgdt_size,
        dm_size,
        norm = "vec![]",
        n_jobs = -1,
        approx_erf = "false"
    )]
    fn from_borders(
        min_lgdt: f64,
        max_lgdt: f64,
        max_abs_dm: f64,
        lgdt_size: usize,
        dm_size: usize,
        norm: Vec<&str>,
        n_jobs: i64,
        approx_erf: bool,
    ) -> Res<Self> {
        let dmdt_f32 = lcdmdt::DmDt::from_lgdt_dm_limits(
            min_lgdt as f32,
            max_lgdt as f32,
            lgdt_size,
            max_abs_dm as f32,
            dm_size,
        );
        let dmdt_f64 =
            lcdmdt::DmDt::from_lgdt_dm_limits(min_lgdt, max_lgdt, lgdt_size, max_abs_dm, dm_size);

        Self::from_dmdts(dmdt_f32, dmdt_f64, norm, n_jobs, approx_erf)
    }

    #[getter]
    fn get_n_jobs(&self) -> usize {
        self.dmdt_f32.n_jobs
    }

    #[setter]
    fn set_n_jobs(&mut self, value: i64) -> Res<()> {
        if value <= 0 {
            Err(Exception::ValueError(
                "cannot set non-positive n_jobs value".to_owned(),
            ))
        } else {
            self.dmdt_f32.n_jobs = value as usize;
            self.dmdt_f64.n_jobs = value as usize;
            Ok(())
        }
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.dmdt_f64.dmdt.shape()
    }

    #[getter]
    fn dt_grid<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        self.dmdt_f64.dmdt.dt_grid.get_borders().to_pyarray(py)
    }

    #[getter]
    fn min_dt(&self) -> f64 {
        self.dmdt_f64.dmdt.dt_grid.get_start()
    }

    #[getter]
    fn max_dt(&self) -> f64 {
        self.dmdt_f64.dmdt.dt_grid.get_end()
    }

    #[getter]
    fn dm_grid<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        self.dmdt_f64.dmdt.dm_grid.get_borders().to_pyarray(py)
    }

    #[getter]
    fn min_dm(&self) -> f64 {
        self.dmdt_f64.dmdt.dm_grid.get_start()
    }

    #[getter]
    fn max_dm(&self) -> f64 {
        self.dmdt_f64.dmdt.dm_grid.get_end()
    }

    /// Total number of observations per each dt interval
    ///
    /// Output takes into account all observation pairs within
    /// [min_dt; max_dt), even if they are not in [min_dm; max_dm)
    ///
    /// Parameters
    /// ----------
    /// t : 1d-ndarray of float
    ///     Time moments, must be sorted
    /// sorted : bool or None, optional
    ///     `True` guarantees that `t` is sorted
    ///
    /// Returns
    /// 1d-array of float
    ///
    #[args(t, sorted = "None")]
    fn count_dt(&self, py: Python, t: GenericFloatArray1, sorted: Option<bool>) -> Res<PyObject> {
        match t {
            GenericFloatArray1::Float32(t) => self.dmdt_f32.py_count_dt(py, t, sorted),
            GenericFloatArray1::Float64(t) => self.dmdt_f64.py_count_dt(py, t, sorted),
        }
    }

    /// Total number of observations per each dt interval
    ///
    /// Output takes into account all observation pairs within
    /// [min_dt; max_dt), even if they are not in [min_dm; max_dm)
    ///
    /// Parameters
    /// ----------
    /// t_ : list of 1d-ndarray of float
    ///     List of arrays, each represents time moments, must be sorted
    /// sorted : bool or None, optional
    ///     `True` guarantees that `t` is sorted
    ///
    /// Returns
    /// 1d-array of float
    ///
    #[args(t_, sorted = "None")]
    fn count_dt_many(
        &self,
        py: Python,
        t_: Vec<GenericFloatArray1>,
        sorted: Option<bool>,
    ) -> Res<PyObject> {
        if t_.is_empty() {
            Err(Exception::ValueError("t_ is empty".to_owned()))
        } else {
            match t_[0] {
                GenericFloatArray1::Float32(_) => self.dmdt_f32.py_count_dt_many(py, t_, sorted),
                GenericFloatArray1::Float64(_) => self.dmdt_f64.py_count_dt_many(py, t_, sorted),
            }
        }
    }

    /// Produces dmdt-map from light curve
    ///
    /// Parameters
    /// ----------
    /// t : 1d-ndarray of float
    ///     Time moments, must be sorted
    /// m : 1d-ndarray of float
    ///     Magnitudes
    /// sorted : bool or None, optional
    ///     `True` guarantees that the light curve is sorted
    ///
    /// Returns
    /// -------
    /// 2d-ndarray of float
    ///
    #[args(t, m, sorted = "None")]
    fn points(
        &self,
        py: Python,
        t: GenericFloatArray1,
        m: GenericFloatArray1,
        sorted: Option<bool>,
    ) -> Res<PyObject> {
        match (t, m) {
            (GenericFloatArray1::Float32(t), GenericFloatArray1::Float32(m)) => {
                self.dmdt_f32.py_points(py, t, m, sorted)
            }
            (GenericFloatArray1::Float64(t), GenericFloatArray1::Float64(m)) => {
                self.dmdt_f64.py_points(py, t, m, sorted)
            }
            _ => Err(Exception::TypeError(
                "t and m must have the same dtype".to_owned(),
            )),
        }
    }

    /// Produces dmdt-map from a collection of light curves
    ///
    /// The method is performed in parallel using `n_jobs` threads
    ///
    /// Parameters
    /// ----------
    /// lcs : list of (ndarray, ndarray)
    ///     List or tuple of tuple pairs (t, m) represented individual light
    ///     curves. All arrays must have the same dtype
    /// sorted : bool or None, optional
    ///     `True` guarantees that all light curves is sorted
    ///
    /// Returns
    /// -------
    /// 3d-ndarray of float
    ///
    #[args(lcs, sorted = "None")]
    fn points_many(
        &self,
        py: Python,
        lcs: Vec<(GenericFloatArray1, GenericFloatArray1)>,
        sorted: Option<bool>,
    ) -> Res<PyObject> {
        if lcs.is_empty() {
            Err(Exception::ValueError("lcs is empty".to_owned()))
        } else {
            match lcs[0].0 {
                GenericFloatArray1::Float32(_) => self.dmdt_f32.py_points_many(py, lcs, sorted),
                GenericFloatArray1::Float64(_) => self.dmdt_f64.py_points_many(py, lcs, sorted),
            }
        }
    }

    /// Reusable iterable yielding dmdt-maps
    ///
    /// The dmdt-maps are produced in parallel using `n_jobs` threads, batches
    /// are being generated in background, so the next batch is started to
    /// generate just after the previous one is yielded. Note that light curves
    /// data are copied, so from the performance point of view it is better to
    /// use `points_many` if you don't need observation dropping
    ///
    /// Parameters
    /// ----------
    /// lcs : list of (ndarray, ndarray)
    ///     List or tuple of tuple pairs (t, m) represented individual light
    ///     curves. All arrays must have the same dtype
    /// sorted : bool or None, optional
    ///     `True` guarantees that all light curves is sorted, default is
    ///     `None`
    /// batch_size : int, optional
    ///     The number of dmdt-maps to yield. The last batch can be smaller.
    ///     Default is 1
    /// yield_index : bool, optional
    ///     Yield a tuple of (indexes, maps) instead of just maps. Could be
    ///     useful when shuffle is `True`. Default is `False`
    /// shuffle : bool, optional
    ///     If `True`, shuffle light curves (not individual observations) on
    ///     each creating of new iterator. Default is `False`
    /// drop_nobs : int or float, optional
    ///     Drop observations from every light curve. If it is a positive
    ///     integer, it is a number of observations to drop. If it is a
    ///     floating point between 0 and 1, it is a part of observation to
    ///     drop. Default is `0`, which means usage of the original data
    /// random_seed : int or None, optional
    ///     Random seed for shuffling and dropping. Default is `None` which
    ///     means random seed
    ///
    /// Returns
    /// -------
    /// Iterable of 3d-ndarray or (1d-ndarray, 3d-ndarray)
    ///
    #[allow(clippy::too_many_arguments)]
    #[args(
        lcs,
        sorted = "None",
        batch_size = 1,
        yield_index = false,
        shuffle = false,
        drop_nobs = "DropNObsType::Int(0)",
        random_seed = "None"
    )]
    fn points_batches(
        &self,
        py: Python,
        lcs: Vec<(GenericFloatArray1, GenericFloatArray1)>,
        sorted: Option<bool>,
        batch_size: usize,
        yield_index: bool,
        shuffle: bool,
        drop_nobs: DropNObsType,
        random_seed: Option<u64>,
    ) -> Res<PyObject> {
        if lcs.is_empty() {
            Err(Exception::ValueError("lcs is empty".to_owned()))
        } else {
            Ok(match lcs[0].0 {
                GenericFloatArray1::Float32(_) => DmDtPointsBatchesF32 {
                    dmdt_batches: Arc::new(self.dmdt_f32.generic_dmdt_points_batches(
                        lcs,
                        sorted,
                        batch_size,
                        yield_index,
                        shuffle,
                        drop_nobs,
                        random_seed,
                    )?),
                }
                .into_py(py),
                GenericFloatArray1::Float64(_) => DmDtPointsBatchesF64 {
                    dmdt_batches: Arc::new(self.dmdt_f64.generic_dmdt_points_batches(
                        lcs,
                        sorted,
                        batch_size,
                        yield_index,
                        shuffle,
                        drop_nobs,
                        random_seed,
                    )?),
                }
                .into_py(py),
            })
        }
    }

    /// Produces smeared dmdt-map from light curve
    ///
    /// Parameters
    /// ----------
    /// t : 1d-ndarray of float
    ///     Time moments, must be sorted
    /// m : 1d-ndarray of float
    ///     Magnitudes
    /// sigma : 1d-ndarray of float
    ///     Uncertainties
    /// sorted : bool or None, optional
    ///     `True` guarantees that the light curve is sorted
    ///
    /// Returns
    /// -------
    /// 2d-array of float
    ///
    #[args(t, m, sigma, sorted = "None")]
    fn gausses(
        &self,
        py: Python,
        t: GenericFloatArray1,
        m: GenericFloatArray1,
        sigma: GenericFloatArray1,
        sorted: Option<bool>,
    ) -> Res<PyObject> {
        match (t, m, sigma) {
            (
                GenericFloatArray1::Float32(t),
                GenericFloatArray1::Float32(m),
                GenericFloatArray1::Float32(sigma),
            ) => self.dmdt_f32.py_gausses(py, t, m, sigma, sorted),
            (
                GenericFloatArray1::Float64(t),
                GenericFloatArray1::Float64(m),
                GenericFloatArray1::Float64(sigma),
            ) => self.dmdt_f64.py_gausses(py, t, m, sigma, sorted),
            _ => Err(Exception::TypeError(
                "t, m and sigma must have the same dtype".to_owned(),
            )),
        }
    }

    /// Produces smeared dmdt-map from a collection of light curves
    ///
    /// The method is performed in parallel using `n_jobs` threads
    ///
    /// Parameters
    /// ----------
    /// lcs : list of (ndarray, ndarray, ndarray)
    ///     List or tuple of tuple pairs (t, m, sigma) represented individual
    ///     light curves. All arrays must have the same dtype
    /// sorted : bool or None, optional
    ///     `True` guarantees that all light curves are sorted
    ///
    /// Returns
    /// -------
    /// 3d-ndarray of float
    ///
    #[args(lcs, sorted = "None")]
    fn gausses_many(
        &self,
        py: Python,
        lcs: Vec<(GenericFloatArray1, GenericFloatArray1, GenericFloatArray1)>,
        sorted: Option<bool>,
    ) -> Res<PyObject> {
        if lcs.is_empty() {
            Err(Exception::ValueError("lcs is empty".to_owned()))
        } else {
            match lcs[0].0 {
                GenericFloatArray1::Float32(_) => self.dmdt_f32.py_gausses_many(py, lcs, sorted),
                GenericFloatArray1::Float64(_) => self.dmdt_f64.py_gausses_many(py, lcs, sorted),
            }
        }
    }

    /// Reusable iterable yielding dmdt-maps
    ///
    /// The dmdt-maps are produced in parallel using `n_jobs` threads, batches
    /// are being generated in background, so the next batch is started to
    /// generate just after the previous one is yielded. Note that light curves
    /// data are copied, so from the performance point of view it is better to
    /// use `gausses_many` if you don't need observation dropping
    ///
    /// Parameters
    /// ----------
    /// lcs : list of (ndarray, ndarray, ndarray)
    ///     List or tuple of tuple pairs (t, m, sigma) represented individual
    ///     light curves. All arrays must have the same dtype
    /// sorted : bool or None, optional
    ///     `True` guarantees that all light curves is sorted, default is
    ///     `None`
    /// batch_size : int, optional
    ///     The number of dmdt-maps to yield. The last batch can be smaller.
    ///     Default is 1
    /// yield_index : bool, optional
    ///     Yield a tuple of (indexes, maps) instead of just maps. Could be
    ///     useful when shuffle is `True`. Default is `False`
    /// shuffle : bool, optional
    ///     If `True`, shuffle light curves (not individual observations) on
    ///     each creating of new iterator. Default is `False`
    /// drop_nobs : int or float, optional
    ///     Drop observations from every light curve. If it is a positive
    ///     integer, it is a number of observations to drop. If it is a
    ///     floating point between 0 and 1, it is a part of observation to
    ///     drop. Default is `0`, which means usage of the original data
    /// random_seed : int or None, optional
    ///     Random seed for shuffling and dropping. Default is `None` which
    ///     means random seed
    ///
    #[allow(clippy::too_many_arguments)]
    #[args(
        lcs,
        sorted = "None",
        batch_size = 1,
        yield_index = false,
        shuffle = false,
        drop_nobs = "DropNObsType::Int(0)",
        random_seed = "None"
    )]
    fn gausses_batches(
        &self,
        py: Python,
        lcs: Vec<(GenericFloatArray1, GenericFloatArray1, GenericFloatArray1)>,
        sorted: Option<bool>,
        batch_size: usize,
        yield_index: bool,
        shuffle: bool,
        drop_nobs: DropNObsType,
        random_seed: Option<u64>,
    ) -> Res<PyObject> {
        if lcs.is_empty() {
            Err(Exception::ValueError("lcs is empty".to_owned()))
        } else {
            match lcs[0].0 {
                GenericFloatArray1::Float32(_) => Ok(DmDtGaussesBatchesF32 {
                    dmdt_batches: Arc::new(self.dmdt_f32.generic_dmdt_gausses_batches(
                        lcs,
                        sorted,
                        batch_size,
                        yield_index,
                        shuffle,
                        drop_nobs,
                        random_seed,
                    )?),
                }
                .into_py(py)),
                GenericFloatArray1::Float64(_) => Ok(DmDtGaussesBatchesF64 {
                    dmdt_batches: Arc::new(self.dmdt_f64.generic_dmdt_gausses_batches(
                        lcs,
                        sorted,
                        batch_size,
                        yield_index,
                        shuffle,
                        drop_nobs,
                        random_seed,
                    )?),
                }
                .into_py(py)),
            }
        }
    }
}
