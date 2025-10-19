//! Macros for implementing SIMD distance functions

/// Macro to implement naive distance functions for a given underlying scalar type
macro_rules! impl_naive {
    ($ty1:ty) => {
        impl Naive for &[$ty1] {
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
        }

        impl Naive for &Vec<$ty1> {
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
        }
    };
}

/// Macro to implement SIMD distance functions for a given SIMD type and underlying scalar type
macro_rules! impl_distances {
    ($name:ident, $ty:ty) => {
        use crate::simd::Naive;

        impl $name {
            /// Calculate the squared distance between two SIMD lane-slices
            pub fn euclidean_inner(a: &[$ty], b: &[$ty]) -> $name {
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
                    return Naive::squared_euclidean(a, b);
                }

                let mut i = 0;
                let mut sum = $name::splat(0.0);
                while a.len() - $name::lanes() >= i {
                    sum += $name::euclidean_inner(&a[i..i + $name::lanes()], &b[i..i + $name::lanes()]);
                    i += $name::lanes();
                }

                let mut sum = sum.horizontal_add();
                if i < a.len() {
                    sum += Naive::squared_euclidean(&a[i..], &b[i..]);
                }
                sum
            }
        }
    };
}
