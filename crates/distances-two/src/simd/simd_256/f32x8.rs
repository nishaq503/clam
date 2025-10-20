//! 256-bit wide SIMD vector of eight f32 values.

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

define_ty!(F32x8, f32, f32, f32, f32, f32, f32, f32, f32);
impl_minimal!(F32x8, f32, 8, x0, x1, x2, x3, x4, x5, x6, x7);

impl F32x8 {
    /// Create a new `F32x8` from a slice.
    ///
    /// # Panics
    ///
    /// Will panic if the slice is not at least 8 elements long.
    pub fn from_slice(slice: &[f32]) -> Self {
        debug_assert!(slice.len() >= Self::lanes());
        Self(
            slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
        )
    }

    /// Sum all lanes of the SIMD vector.
    pub fn horizontal_add(self) -> f32 {
        self.0 + self.1 + self.2 + self.3 + self.4 + self.5 + self.6 + self.7
    }
}

impl_op8!(Mul, mul, F32x8, *);
impl_op8!(assn MulAssign, mul_assign, F32x8, *=);
impl_op8!(Div, div, F32x8, /);
impl_op8!(assn DivAssign, div_assign, F32x8, /=);
impl_op8!(Add, add, F32x8, +);
impl_op8!(assn AddAssign, add_assign, F32x8, +=);
impl_op8!(Sub, sub, F32x8, -);
impl_op8!(assn SubAssign, sub_assign, F32x8, -=);

impl_distances!(F32x8, f32);

impl super::super::SIMD for &[f32] {
    type Output = f32;

    fn squared_euclidean(self, other: Self) -> Self::Output {
        if self.len() < 8 {
            super::super::Naive::squared_euclidean(self, other)
        } else {
            F32x8::squared_euclidean(self, other)
        }
    }

    fn euclidean(self, other: Self) -> Self::Output {
        super::super::SIMD::squared_euclidean(self, other).sqrt()
    }
}

impl super::super::SIMD for &Vec<f32> {
    type Output = f32;

    fn squared_euclidean(self, other: Self) -> Self::Output {
        if self.len() < 8 {
            super::super::Naive::squared_euclidean(self, other)
        } else {
            F32x8::squared_euclidean(self, other)
        }
    }

    fn euclidean(self, other: Self) -> Self::Output {
        super::super::SIMD::squared_euclidean(self, other).sqrt()
    }
}
