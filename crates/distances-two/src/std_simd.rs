//! SIMD accelerated distance functions using the `core::simd` module in nightly Rust.

#![allow(clippy::similar_names)]

use core::ops::{Add, Mul, Sub};
use core::simd::{LaneCount, SimdElement, SupportedLaneCount, prelude::*};

/// Computes the Squared Euclidean distance between two vectors using SIMD with a specified number of lanes.
pub fn euclidean_sq<T, S, const LANES: usize>(a: S, b: S) -> T
where
    T: num_traits::Float + SimdElement,
    S: AsRef<[T]>,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: Sub<Output = Simd<T, LANES>> + Mul<Output = Simd<T, LANES>> + Add<Output = Simd<T, LANES>>,
{
    let a = a.as_ref();
    let b = b.as_ref();
    debug_assert_eq!(a.len(), b.len());

    let mut a_chunks = a.chunks_exact(LANES);
    let mut b_chunks = b.chunks_exact(LANES);

    let sum_simd = a_chunks
        .by_ref()
        .map(Simd::<T, LANES>::from_slice)
        .zip(b_chunks.by_ref().map(Simd::<T, LANES>::from_slice))
        .fold(Simd::splat(T::zero()), |acc, (x, y)| {
            let diff = x - y;
            acc + diff * diff
        });

    let sum_rem = a_chunks.remainder().iter().zip(b_chunks.remainder().iter()).fold(T::zero(), |acc, (&x, &y)| {
        let diff = x - y;
        acc + diff * diff
    });

    sum_simd.as_array().iter().fold(sum_rem, |acc, &v| acc + v)
}

/// Computes the Euclidean distance between two vectors using SIMD with a specified number of lanes.
pub fn euclidean<T, S, const LANES: usize>(a: S, b: S) -> T
where
    T: num_traits::Float + SimdElement,
    S: AsRef<[T]>,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: core::ops::Sub<Output = Simd<T, LANES>> + core::ops::Mul<Output = Simd<T, LANES>> + core::ops::Add<Output = Simd<T, LANES>>,
{
    euclidean_sq::<_, _, LANES>(a, b).sqrt()
}

/// Computes the Dot Product between two vectors using SIMD with a specified number of lanes.
pub fn dot_product<T, S, const LANES: usize>(a: S, b: S) -> T
where
    T: num_traits::Float + SimdElement,
    S: AsRef<[T]>,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: core::ops::Sub<Output = Simd<T, LANES>> + core::ops::Mul<Output = Simd<T, LANES>> + core::ops::Add<Output = Simd<T, LANES>>,
{
    let a = a.as_ref();
    let b = b.as_ref();
    debug_assert_eq!(a.len(), b.len());

    let mut a_chunks = a.chunks_exact(LANES);
    let mut b_chunks = b.chunks_exact(LANES);

    let sum_simd = a_chunks
        .by_ref()
        .map(Simd::<T, LANES>::from_slice)
        .zip(b_chunks.by_ref().map(Simd::<T, LANES>::from_slice))
        .fold(Simd::splat(T::zero()), |acc, (x, y)| acc + x * y);

    let sum_rem = a_chunks
        .remainder()
        .iter()
        .zip(b_chunks.remainder().iter())
        .fold(T::zero(), |acc, (&x, &y)| acc + x * y);

    sum_simd.as_array().iter().fold(sum_rem, |acc, &v| acc + v)
}

/// Computes the Squared L2 norm of a vector using SIMD with a specified number of lanes.
pub fn norm_l2_sq<T, S, const LANES: usize>(a: S) -> T
where
    T: num_traits::Float + SimdElement,
    S: AsRef<[T]>,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: core::ops::Sub<Output = Simd<T, LANES>> + core::ops::Mul<Output = Simd<T, LANES>> + core::ops::Add<Output = Simd<T, LANES>>,
{
    let mut a_chunks = a.as_ref().chunks_exact(LANES);

    let sum_simd = a_chunks
        .by_ref()
        .map(Simd::<T, LANES>::from_slice)
        .fold(Simd::splat(T::zero()), |acc, x| acc + x * x);

    let sum_rem = a_chunks.remainder().iter().fold(T::zero(), |acc, &x| acc + x * x);

    sum_simd.as_array().iter().fold(sum_rem, |acc, &v| acc + v)
}

/// Computes the L2 norm of a vector using SIMD with a specified number of lanes.
pub fn norm_l2<T, S, const LANES: usize>(a: S) -> T
where
    T: num_traits::Float + SimdElement,
    S: AsRef<[T]>,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: core::ops::Sub<Output = Simd<T, LANES>> + core::ops::Mul<Output = Simd<T, LANES>> + core::ops::Add<Output = Simd<T, LANES>>,
{
    norm_l2_sq::<_, _, LANES>(a).sqrt()
}

/// Computes the Cosine distance between two vectors using SIMD with a specified number of lanes.
pub fn cosine<T, S, const LANES: usize>(a: S, b: S) -> T
where
    T: num_traits::Float + SimdElement,
    S: AsRef<[T]>,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: core::ops::Sub<Output = Simd<T, LANES>> + core::ops::Mul<Output = Simd<T, LANES>> + core::ops::Add<Output = Simd<T, LANES>>,
{
    T::one() - cosine_similarity(a, b)
}

/// Computes the Cosine similarity between two vectors using SIMD with a specified number of lanes.
pub fn cosine_similarity<T, S, const LANES: usize>(a: S, b: S) -> T
where
    T: num_traits::Float + SimdElement,
    S: AsRef<[T]>,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: core::ops::Sub<Output = Simd<T, LANES>> + core::ops::Mul<Output = Simd<T, LANES>> + core::ops::Add<Output = Simd<T, LANES>>,
{
    let a = a.as_ref();
    let b = b.as_ref();
    debug_assert_eq!(a.len(), b.len());

    let mut a_chunks = a.chunks_exact(LANES);
    let mut b_chunks = b.chunks_exact(LANES);

    let (ab_sum_simd, aa_sum_simd, bb_sum_simd) = a_chunks
        .by_ref()
        .map(Simd::<T, LANES>::from_slice)
        .zip(b_chunks.by_ref().map(Simd::<T, LANES>::from_slice))
        .fold(
            (Simd::splat(T::zero()), Simd::splat(T::zero()), Simd::splat(T::zero())),
            |(ab_acc, aa_acc, bb_acc), (x, y)| (ab_acc + x * y, aa_acc + x * x, bb_acc + y * y),
        );

    let (ab_sum_rem, aa_sum_rem, bb_sum_rem) = a_chunks
        .remainder()
        .iter()
        .zip(b_chunks.remainder().iter())
        .fold((T::zero(), T::zero(), T::zero()), |(ab_acc, aa_acc, bb_acc), (&x, &y)| {
            (ab_acc + x * y, aa_acc + x * x, bb_acc + y * y)
        });

    let ab = ab_sum_simd.as_array().iter().fold(ab_sum_rem, |acc, &v| acc + v);
    if ab.is_zero() {
        T::zero()
    } else {
        let aa = aa_sum_simd.as_array().iter().fold(aa_sum_rem, |acc, &v| acc + v);
        let bb = bb_sum_simd.as_array().iter().fold(bb_sum_rem, |acc, &v| acc + v);
        ab / (aa * bb).sqrt()
    }
}

/// Computes the Cosine distance between two vectors that have unit l2 norm using SIMD with a specified number of lanes.
pub fn cosine_normalized<T, S, const LANES: usize>(a: S, b: S) -> T
where
    T: num_traits::Float + SimdElement,
    S: AsRef<[T]>,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: core::ops::Sub<Output = Simd<T, LANES>> + core::ops::Mul<Output = Simd<T, LANES>> + core::ops::Add<Output = Simd<T, LANES>>,
{
    T::one() - dot_product(a, b)
}

/// Computes the Cosine similarity between two vectors that have unit l2 norm using SIMD with a specified number of lanes.
pub fn cosine_similarity_normalized<T, S, const LANES: usize>(a: S, b: S) -> T
where
    T: num_traits::Float + SimdElement,
    S: AsRef<[T]>,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: core::ops::Sub<Output = Simd<T, LANES>> + core::ops::Mul<Output = Simd<T, LANES>> + core::ops::Add<Output = Simd<T, LANES>>,
{
    dot_product(a, b)
}
