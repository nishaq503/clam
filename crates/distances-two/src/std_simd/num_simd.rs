//! SIMD accelerated numerical operations.

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};
use core::simd::{LaneCount, SimdElement, SupportedLaneCount, prelude::*};

use num_traits::{One, Zero};

/// A trait for numeric SIMD elements.
pub trait NumSimdElement: SimdElement + Copy + num_traits::Num + num_traits::NumAssignOps {}

impl<T> NumSimdElement for T where T: SimdElement + Copy + num_traits::Num + num_traits::NumAssignOps {}

/// A SIMD wrapper for numeric types.
pub struct NumSimd<T, const LANES: usize>
where
    T: NumSimdElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    /// The underlying SIMD data.
    data: Simd<T, LANES>,
}

impl<T, const LANES: usize> NumSimd<T, LANES>
where
    T: NumSimdElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    /// Creates a new `NumSimd` from a SIMD vector.
    pub fn new(data: Simd<T, LANES>) -> Self {
        Self { data }
    }

    /// Returns the underlying SIMD data.
    pub fn data(&self) -> &Simd<T, LANES> {
        &self.data
    }

    /// Creates a `NumSimd` from a slice.
    pub fn from_slice(slice: &[T]) -> Self {
        Self {
            data: Simd::<T, LANES>::from_slice(slice),
        }
    }

    /// Creates a `NumSimd` by splatting a single value.
    pub fn splat(value: T) -> Self {
        Self {
            data: Simd::<T, LANES>::splat(value),
        }
    }
}

impl<T, const LANES: usize> PartialEq for NumSimd<T, LANES>
where
    T: NumSimdElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn eq(&self, other: &Self) -> bool {
        self.data.eq(&other.data)
    }
}

impl<T, const LANES: usize> Add for NumSimd<T, LANES>
where
    T: NumSimdElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: Add<Output = Simd<T, LANES>>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            data: self.data + rhs.data,
        }
    }
}

impl<T, const LANES: usize> Sub for NumSimd<T, LANES>
where
    T: NumSimdElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: Sub<Output = Simd<T, LANES>>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            data: self.data - rhs.data,
        }
    }
}

impl<T, const LANES: usize> Mul for NumSimd<T, LANES>
where
    T: NumSimdElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: Mul<Output = Simd<T, LANES>>,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            data: self.data * rhs.data,
        }
    }
}

impl<T, const LANES: usize> Div for NumSimd<T, LANES>
where
    T: NumSimdElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: Div<Output = Simd<T, LANES>>,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self {
            data: self.data / rhs.data,
        }
    }
}

impl<T, const LANES: usize> Rem for NumSimd<T, LANES>
where
    T: NumSimdElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: Rem<Output = Simd<T, LANES>>,
{
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        Self {
            data: self.data % rhs.data,
        }
    }
}

impl<T, const LANES: usize> AddAssign for NumSimd<T, LANES>
where
    T: NumSimdElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        self.data += rhs.data;
    }
}

impl<T, const LANES: usize> SubAssign for NumSimd<T, LANES>
where
    T: NumSimdElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: SubAssign,
{
    fn sub_assign(&mut self, rhs: Self) {
        self.data -= rhs.data;
    }
}

impl<T, const LANES: usize> MulAssign for NumSimd<T, LANES>
where
    T: NumSimdElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: MulAssign,
{
    fn mul_assign(&mut self, rhs: Self) {
        self.data *= rhs.data;
    }
}

impl<T, const LANES: usize> DivAssign for NumSimd<T, LANES>
where
    T: NumSimdElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: DivAssign,
{
    fn div_assign(&mut self, rhs: Self) {
        self.data /= rhs.data;
    }
}

impl<T, const LANES: usize> RemAssign for NumSimd<T, LANES>
where
    T: NumSimdElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: RemAssign,
{
    fn rem_assign(&mut self, rhs: Self) {
        self.data %= rhs.data;
    }
}

impl<T, const LANES: usize> Zero for NumSimd<T, LANES>
where
    T: NumSimdElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: Zero,
{
    fn zero() -> Self {
        Self {
            data: Simd::<T, LANES>::splat(T::zero()),
        }
    }

    fn is_zero(&self) -> bool {
        self.data.eq(&Simd::<T, LANES>::splat(T::zero()))
    }
}

impl<T, const LANES: usize> One for NumSimd<T, LANES>
where
    T: NumSimdElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: One,
{
    fn one() -> Self {
        Self {
            data: Simd::<T, LANES>::splat(T::one()),
        }
    }
}
