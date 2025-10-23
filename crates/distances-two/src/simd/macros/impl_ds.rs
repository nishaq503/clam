//! Macros for implementing SIMD distance functions

/// Macro to implement naive distance functions for a given underlying scalar type
macro_rules! impl_naive {
    ($ty1:ty) => {
        impl crate::simd::Naive for &[$ty1] {
            type Output = $ty1;

            fn squared_euclidean(self, other: Self) -> Self::Output {
                assert_eq!(self.len(), other.len());

                let mut sum = 0.0;
                for i in 0..self.len() {
                    let d = self[i] - other[i];
                    sum += (d * d);
                }
                sum
            }

            fn dot_product(self, other: Self) -> Self::Output {
                assert_eq!(self.len(), other.len());

                let mut sum = 0.0;
                for i in 0..self.len() {
                    sum += self[i] * other[i];
                }
                sum
            }
        }

        impl crate::simd::Naive for &Vec<$ty1> {
            type Output = $ty1;

            fn squared_euclidean(self, other: Self) -> $ty1 {
                assert_eq!(self.len(), other.len());

                let mut sum = 0.0;
                for i in 0..self.len() {
                    let d = self[i] - other[i];
                    sum += (d * d);
                }
                sum
            }

            fn dot_product(self, other: Self) -> Self::Output {
                assert_eq!(self.len(), other.len());

                let mut sum = 0.0;
                for i in 0..self.len() {
                    sum += self[i] * other[i];
                }
                sum
            }
        }
    };
}

/// Macro to implement SIMD distance functions for a given SIMD type and underlying scalar type
macro_rules! impl_distances {
    ($name:ident, $ty:ty) => {
        impl $name {
            /// Calculate the squared distance between two SIMD lane-slices
            fn euclidean_inner(a: &[$ty], b: &[$ty]) -> $name {
                let i = $name::from_slice(a);
                let j = $name::from_slice(b);
                let u = i - j;
                u * u
            }

            /// Calculate the squared euclidean distance between two slices of
            /// equal length, using auto-vectorized SIMD primitives
            pub fn squared_euclidean(a: &[$ty], b: &[$ty]) -> $ty {
                assert_eq!(a.len(), b.len());
                if a.len() < $name::lanes() {
                    return crate::simd::Naive::squared_euclidean(a, b);
                }

                let mut i = 0;
                let mut sum = $name::splat(0.0);
                while a.len() - $name::lanes() >= i {
                    sum += $name::euclidean_inner(&a[i..i + $name::lanes()], &b[i..i + $name::lanes()]);
                    i += $name::lanes();
                }

                let mut sum = sum.horizontal_add();
                if i < a.len() {
                    sum += crate::simd::Naive::squared_euclidean(&a[i..], &b[i..]);
                }
                sum
            }

            fn dot_inner(a: &[$ty], b: &[$ty]) -> $name {
                let i = $name::from_slice(a);
                let j = $name::from_slice(b);
                i * j
            }

            pub fn dot_product(a: &[$ty], b: &[$ty]) -> $ty {
                assert_eq!(a.len(), b.len());
                if a.len() < $name::lanes() {
                    return crate::simd::Naive::dot_product(a, b);
                }

                let mut i = 0;
                let mut sum = $name::splat(0.0);
                while a.len() - $name::lanes() >= i {
                    sum += $name::dot_inner(&a[i..i + $name::lanes()], &b[i..i + $name::lanes()]);
                    i += $name::lanes();
                }

                let mut sum = sum.horizontal_add();
                if i < a.len() {
                    sum += crate::simd::Naive::dot_product(&a[i..], &b[i..]);
                }
                sum
            }
        }
    };
}

/// Macro to implement the SIMD trait for a given SIMD type, underlying scalar type, and array type
macro_rules! impl_simd {
    ($name:ident, $ty:ty, $arr:ty) => {
        impl crate::simd::SIMD for $arr {
            type Output = $ty;

            fn squared_euclidean(self, other: Self) -> Self::Output {
                debug_assert_eq!(self.len(), other.len());

                let mut a_chunks = self.chunks_exact($name::lanes());
                let mut b_chunks = other.chunks_exact($name::lanes());

                let sum = a_chunks
                    .by_ref()
                    .map($name::from_slice)
                    .zip(b_chunks.by_ref().map($name::from_slice))
                    .map(|(a, b)| {
                        let diff = a - b;
                        diff * diff
                    })
                    .fold($name::splat(0.0), |acc, x| acc + x)
                    .horizontal_add();

                let rem = crate::vectors::euclidean_sq(&a_chunks.remainder(), &b_chunks.remainder());

                sum + rem
            }

            fn euclidean(self, other: Self) -> Self::Output {
                crate::simd::SIMD::squared_euclidean(self, other).sqrt()
            }

            fn dot_product(self, other: Self) -> Self::Output {
                debug_assert_eq!(self.len(), other.len());

                let mut a_chunks = self.chunks_exact($name::lanes());
                let mut b_chunks = other.chunks_exact($name::lanes());

                let sum = a_chunks
                    .by_ref()
                    .map($name::from_slice)
                    .zip(b_chunks.by_ref().map($name::from_slice))
                    .map(|(a, b)| a * b)
                    .fold($name::splat(0.0), |acc, x| acc + x)
                    .horizontal_add();

                let rem = crate::vectors::dot_product(&a_chunks.remainder(), &b_chunks.remainder());

                sum + rem
            }
        }
    };
}
