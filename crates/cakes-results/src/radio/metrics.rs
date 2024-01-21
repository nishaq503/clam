//! Distance metrics for the `RadioML` data.

use core::{
    fmt::{Debug, Display},
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign},
};

use distances::Number;
use num_complex::Complex32;

/// Returns the Dynamic Time Warping distance between the two given time series.
///
/// # Arguments
///
/// * `x` - The first time series.
/// * `y` - The second time series.
#[allow(clippy::ptr_arg)]
pub fn dtw(x: &Vec<C32>, y: &Vec<C32>) -> f32 {
    let mut matrix = vec![vec![f32::INFINITY; x.len() + 1]; y.len() + 1];
    matrix[0][0] = 0.0;

    for i in 1..=x.len() {
        for j in 1..=y.len() {
            let cost = (x[i - 1] - y[j - 1]).norm_sq();
            matrix[i][j] = cost + min3(matrix[i - 1][j], matrix[i][j - 1], matrix[i - 1][j - 1]);
        }
    }

    matrix[x.len()][y.len()].sqrt()
}

/// Returns the smallest of the three given numbers.
fn min3(a: f32, b: f32, c: f32) -> f32 {
    a.min(b).min(c)
}

/// A complex number with 32-bit floating point components in cartesian form.
#[derive(Default, Debug, Clone, Copy)]
pub struct C32(Complex32);

impl C32 {
    /// Creates a new complex number with the given components.
    #[allow(dead_code)]
    pub const fn new(re: f32, im: f32) -> Self {
        Self(Complex32::new(re, im))
    }

    /// Converts the given complex number to a `C32`.
    pub const fn from_complex(c: Complex32) -> Self {
        Self(c)
    }

    /// Returns the norm of the complex number.
    #[allow(dead_code)]
    pub fn norm(self) -> f32 {
        self.0.norm()
    }

    /// Returns the squared norm of the complex number.
    pub fn norm_sq(self) -> f32 {
        self.0.norm_sqr()
    }

    /// Normalizes the given signal to have unit energy.
    pub fn normalize(signal: Vec<Self>) -> Vec<Self> {
        let norm = signal.iter().map(|x| x.norm_sq()).sum::<f32>();
        signal
            .into_iter()
            .map(|x| Self::new(x.0.re / norm, x.0.im / norm))
            .collect()
    }
}

impl Display for C32 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} + i * {}", self.0.re, self.0.im)
    }
}

impl PartialEq for C32 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl PartialOrd for C32 {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.0.norm().partial_cmp(&other.0.norm())
    }
}

impl Rem for C32 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        Self(self.0 % rhs.0)
    }
}

impl RemAssign for C32 {
    fn rem_assign(&mut self, rhs: Self) {
        self.0 %= rhs.0;
    }
}

impl Div for C32 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}

impl DivAssign for C32 {
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}

impl Mul for C32 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl MulAssign for C32 {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl Sub for C32 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl SubAssign for C32 {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl Add for C32 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign for C32 {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl Sum for C32 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |a, b| a + b)
    }
}

impl Number for C32 {
    fn zero() -> Self {
        Self(Complex32::new(0.0, 0.0))
    }

    fn one() -> Self {
        Self(Complex32::new(1.0, 0.0))
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        (self * a) + b
    }

    fn mul_add_assign(&mut self, a: Self, b: Self) {
        *self = self.mul_add(a, b);
    }

    /// Uses the given number as the real part of a complex number.
    fn from<T: Number>(n: T) -> Self {
        Self(Complex32::new(n.as_f32(), 0.0))
    }

    /// Returns the real part of the complex number.
    fn as_f32(self) -> f32 {
        self.0.re
    }

    /// Returns the real part of the complex number.
    fn as_f64(self) -> f64 {
        self.0.re.as_f64()
    }

    /// Returns the real part of the complex number.
    fn as_u64(self) -> u64 {
        self.0.re.as_u64()
    }

    /// Returns the real part of the complex number.
    fn as_i64(self) -> i64 {
        self.0.re.as_i64()
    }

    /// Returns the complex number without any change.
    fn abs(self) -> Self {
        self
    }

    fn abs_diff(self, other: Self) -> Self {
        self - other
    }

    fn powi(self, exp: i32) -> Self {
        Self(self.0.powi(exp))
    }

    fn num_bytes() -> usize {
        8
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        let mut re_bytes = [0; 4];
        let mut im_bytes = [0; 4];
        re_bytes.copy_from_slice(&bytes[..4]);
        im_bytes.copy_from_slice(&bytes[4..]);
        let re = f32::from_le_bytes(re_bytes);
        let im = f32::from_le_bytes(im_bytes);
        Self(Complex32::new(re, im))
    }

    fn to_le_bytes(self) -> Vec<u8> {
        self.0
            .re
            .to_le_bytes()
            .into_iter()
            .chain(self.0.im.to_le_bytes())
            .collect()
    }

    fn from_be_bytes(bytes: &[u8]) -> Self {
        let mut re_bytes = [0; 4];
        let mut im_bytes = [0; 4];
        re_bytes.copy_from_slice(&bytes[..4]);
        im_bytes.copy_from_slice(&bytes[4..]);
        let re = f32::from_be_bytes(re_bytes);
        let im = f32::from_be_bytes(im_bytes);
        Self(Complex32::new(re, im))
    }

    fn to_be_bytes(self) -> Vec<u8> {
        self.0
            .re
            .to_be_bytes()
            .into_iter()
            .chain(self.0.im.to_be_bytes())
            .collect()
    }

    fn epsilon() -> Self {
        Self(Complex32::new(f32::EPSILON, 0.0))
    }

    fn next_random<R: rand::Rng>(rng: &mut R) -> Self {
        Self(Complex32::new(rng.gen(), rng.gen()))
    }
}
