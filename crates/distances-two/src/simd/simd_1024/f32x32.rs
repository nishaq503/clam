//! 1024-bit SIMD vector of thirty-two f32 values

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

define_type!(
    F32x32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
    f32, f32, f32
);
impl_type!(
    F32x32, f32, 32, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28,
    x29, x30, x31
);

impl F32x32 {
    /// Create a new `F32x32` from a slice.
    ///
    /// # Panics
    ///
    /// Will panic if the slice is not at least 32 elements long.
    pub fn from_slice(slice: &[f32]) -> Self {
        debug_assert!(slice.len() >= Self::lanes());
        Self(
            slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7], slice[8], slice[9], slice[10], slice[11], slice[12], slice[13],
            slice[14], slice[15], slice[16], slice[17], slice[18], slice[19], slice[20], slice[21], slice[22], slice[23], slice[24], slice[25], slice[26],
            slice[27], slice[28], slice[29], slice[30], slice[31],
        )
    }

    /// Sum all lanes of the SIMD vector.
    pub fn horizontal_add(self) -> f32 {
        self.0
            + self.1
            + self.2
            + self.3
            + self.4
            + self.5
            + self.6
            + self.7
            + self.8
            + self.9
            + self.10
            + self.11
            + self.12
            + self.13
            + self.14
            + self.15
            + self.16
            + self.17
            + self.18
            + self.19
            + self.20
            + self.21
            + self.22
            + self.23
            + self.24
            + self.25
            + self.26
            + self.27
            + self.28
            + self.29
            + self.30
            + self.31
    }
}

impl_op32!(Mul, mul, F32x32, *);
impl_op32!(assn MulAssign, mul_assign, F32x32, *=);
impl_op32!(Div, div, F32x32, /);
impl_op32!(assn DivAssign, div_assign, F32x32, /=);
impl_op32!(Add, add, F32x32, +);
impl_op32!(assn AddAssign, add_assign, F32x32, +=);
impl_op32!(Sub, sub, F32x32, -);
impl_op32!(assn SubAssign, sub_assign, F32x32, -=);

impl_simd!(F32x32, f32, &[f32]);
impl_simd!(F32x32, f32, &Vec<f32>);
