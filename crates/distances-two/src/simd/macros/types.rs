//! Macros for implementing SIMD types

/// Macro for defining a SIMD type
macro_rules! define_type {
    ($id:ident, $($elem_tys:ident),+) => {
        #[derive(Copy, Clone, PartialEq, Debug)]
        pub struct $id($($elem_tys),*);
    }
}

/// Macros for implementing minimal SIMD type functionality
macro_rules! impl_type {
    ($id:ident, $elem_ty:ty, $n_lanes:expr, $($elem_name:ident),+) => {
        impl $id {
            #[inline]
            pub const fn lanes() -> usize {
                $n_lanes
            }

            pub const fn splat(value: $elem_ty) -> Self {
                $id($({
                    // this just allows us to repeat over the elements
                    #[expect(non_camel_case_types)]
                    struct $elem_name;
                    value
                }),*)
            }
        }
    };
}
