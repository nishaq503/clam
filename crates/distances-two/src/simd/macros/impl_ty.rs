//! Macros for implementing SIMD types

/// Macro for defining a SIMD type
macro_rules! define_ty {
    ($id:ident, $($elem_tys:ident),+) => {
        #[derive(Copy, Clone, PartialEq, Debug)]
        pub struct $id($($elem_tys),*);
    }
}

/// Macros for implementing minimal SIMD type functionality
macro_rules! impl_minimal {
    ($id:ident, $elem_ty:ident, $elem_count:expr, $($elem_name:ident),+) => {
        impl $id {
            #[inline]
            pub const fn new($($elem_name: $elem_ty),*) -> Self {
                $id($($elem_name),*)
            }

            #[inline]
            pub const fn lanes() -> usize {
                $elem_count
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
