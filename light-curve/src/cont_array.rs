use ndarray::{Array1, ArrayBase, ArrayView1, CowRepr, Ix1, OwnedRepr, RawData, ViewRepr};
use num_traits::Zero;
use numpy::Element;

pub struct ContArrayBase<S: RawData>(pub ArrayBase<S, Ix1>);

pub type ContArray<T> = ContArrayBase<OwnedRepr<T>>;
pub type ContArrayView<'a, T> = ContArrayBase<ViewRepr<&'a T>>;
pub type ContCowArray<'a, T> = ContArrayBase<CowRepr<'a, T>>;

impl<S> ContArrayBase<S>
where
    S: ndarray::Data,
    S::Elem: Clone,
{
    pub fn as_slice(&self) -> &[S::Elem] {
        self.0.as_slice().unwrap()
    }

    pub fn into_owned(self) -> ContArray<S::Elem> {
        ContArrayBase::<OwnedRepr<S::Elem>>(self.0.into_owned())
    }
}

impl<'a, T> ContCowArray<'a, T>
where
    T: Element + Zero + Copy,
{
    pub fn from_view(a: ArrayView1<'a, T>, required: bool) -> Self {
        if required || a.is_standard_layout() {
            a.into()
        } else {
            // TODO: Use the same broadcast trick as in light-curve-feature to speed-up this
            Self(Array1::zeros(a.len()).into())
        }
    }
}

impl<'a, T> From<Array1<T>> for ContArray<T>
where
    T: Element + Copy,
{
    fn from(a: Array1<T>) -> Self {
        if a.is_standard_layout() {
            Self(a)
        } else {
            let owned = a.iter().copied().collect::<Vec<_>>();
            Self(Array1::from_vec(owned))
        }
    }
}

impl<'a, T> From<ArrayView1<'a, T>> for ContArray<T>
where
    T: Element + Copy,
{
    fn from(a: ArrayView1<'a, T>) -> Self {
        let cow: ContCowArray<_> = a.into();
        cow.into_owned()
    }
}

impl<'a, T> From<ArrayView1<'a, T>> for ContCowArray<'a, T>
where
    T: Element + Copy,
{
    fn from(a: ArrayView1<'a, T>) -> Self {
        if a.is_standard_layout() {
            Self(a.into())
        } else {
            let owned_vec = a.iter().copied().collect::<Vec<_>>();
            let array = Array1::from_vec(owned_vec);
            let cow = array.into();
            Self(cow)
        }
    }
}

impl<'a, T> From<ContArray<T>> for ContCowArray<'a, T> {
    fn from(a: ContArray<T>) -> Self {
        Self(a.0.into())
    }
}

impl<'a, T> From<ContArrayView<'a, T>> for ContCowArray<'a, T> {
    fn from(a: ContArrayView<'a, T>) -> Self {
        Self(a.0.into())
    }
}
