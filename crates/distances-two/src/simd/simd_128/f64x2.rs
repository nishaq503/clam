//! 128-bit SIMD vector of four f32 values

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

define_ty!(F64x2, f64, f64);
impl_minimal!(F64x2, f64, 2, x0, x1);

impl F64x2 {
    /// Create a new `F64x2` from a slice.
    ///
    /// # Panics
    ///
    /// Will panic if the slice is not at least 2 elements long.
    pub fn from_slice(slice: &[f64]) -> Self {
        debug_assert!(slice.len() >= Self::lanes());
        Self(slice[0], slice[1])
    }

    /// Sum all lanes of the SIMD vector.
    pub fn horizontal_add(self) -> f64 {
        self.0 + self.1
    }
}

impl_op2!(Mul, mul, F64x2, *);
impl_op2!(assn MulAssign, mul_assign, F64x2, *=);
impl_op2!(Div, div, F64x2, /);
impl_op2!(assn DivAssign, div_assign, F64x2, /=);
impl_op2!(Add, add, F64x2, +);
impl_op2!(assn AddAssign, add_assign, F64x2, +=);
impl_op2!(Sub, sub, F64x2, -);
impl_op2!(assn SubAssign, sub_assign, F64x2, -=);

impl_distances!(F64x2, f64);

impl super::super::SIMD for &[f64] {
    type Output = f64;

    fn squared_euclidean(self, other: Self) -> Self::Output {
        if self.len() < 2 {
            super::super::Naive::squared_euclidean(self, other)
        } else {
            F64x2::squared_euclidean(self, other)
        }
    }

    fn euclidean(self, other: Self) -> Self::Output {
        super::super::SIMD::squared_euclidean(self, other).sqrt()
    }
}

impl super::super::SIMD for &Vec<f64> {
    type Output = f64;

    fn squared_euclidean(self, other: Self) -> Self::Output {
        if self.len() < 2 {
            super::super::Naive::squared_euclidean(self, other)
        } else {
            F64x2::squared_euclidean(self, other)
        }
    }

    fn euclidean(self, other: Self) -> Self::Output {
        super::super::SIMD::squared_euclidean(self, other).sqrt()
    }
}
