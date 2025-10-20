//! Macros for implementing SIMD operations

/// Implement operations for 2-lane SIMD types
macro_rules! impl_op2 {
    ($trait:ident, $fn:ident, $typ:ty, $op:tt) => {
        impl $trait for $typ {
            type Output = $typ;

            fn $fn(self, rhs: Self) -> Self::Output {
                Self(
                    self.0 $op rhs.0,
                    self.1 $op rhs.1,
                )
            }
        }
    };

    (assn $trait:ident, $fn:ident, $typ:ty, $op:tt) => {
        impl $trait for $typ {
            fn $fn(&mut self, rhs: Self) {
                    self.0 $op rhs.0;
                    self.1 $op rhs.1;
            }
        }
    };
}

/// Implement operations for 4-lane SIMD types
macro_rules! impl_op4 {
    ($trait:ident, $fn:ident, $typ:ty, $op:tt) => {
        impl $trait for $typ {
            type Output = $typ;

            fn $fn(self, rhs: Self) -> Self::Output {
                Self(
                    self.0 $op rhs.0,
                    self.1 $op rhs.1,
                    self.2 $op rhs.2,
                    self.3 $op rhs.3,
                )
            }
        }
    };

    (assn $trait:ident, $fn:ident, $typ:ty, $op:tt) => {
        impl $trait for $typ {
            fn $fn(&mut self, rhs: Self) {
                    self.0 $op rhs.0;
                    self.1 $op rhs.1;
                    self.2 $op rhs.2;
                    self.3 $op rhs.3;
            }
        }
    };
}

/// Implement operations for 8-lane SIMD types
macro_rules! impl_op8 {
    ($trait:ident, $fn:ident, $typ:ty, $op:tt) => {
        impl $trait for $typ {
            type Output = $typ;

            fn $fn(self, rhs: Self) -> Self::Output {
                Self(
                    self.0 $op rhs.0,
                    self.1 $op rhs.1,
                    self.2 $op rhs.2,
                    self.3 $op rhs.3,
                    self.4 $op rhs.4,
                    self.5 $op rhs.5,
                    self.6 $op rhs.6,
                    self.7 $op rhs.7,
                )
            }
        }
    };

    (assn $trait:ident, $fn:ident, $typ:ty, $op:tt) => {
        impl $trait for $typ {
            fn $fn(&mut self, rhs: Self) {
                    self.0 $op rhs.0;
                    self.1 $op rhs.1;
                    self.2 $op rhs.2;
                    self.3 $op rhs.3;
                    self.4 $op rhs.4;
                    self.5 $op rhs.5;
                    self.6 $op rhs.6;
                    self.7 $op rhs.7;
            }
        }
    };
}

/// Implement operations for 16-lane SIMD types
macro_rules! impl_op16 {
    ($trait:ident, $fn:ident, $typ:ty, $op:tt) => {
        impl $trait for $typ {
            type Output = $typ;

            fn $fn(self, rhs: Self) -> Self::Output {
                Self(
                    self.0 $op rhs.0,
                    self.1 $op rhs.1,
                    self.2 $op rhs.2,
                    self.3 $op rhs.3,
                    self.4 $op rhs.4,
                    self.5 $op rhs.5,
                    self.6 $op rhs.6,
                    self.7 $op rhs.7,
                    self.8 $op rhs.8,
                    self.9 $op rhs.9,
                    self.10 $op rhs.10,
                    self.11 $op rhs.11,
                    self.12 $op rhs.12,
                    self.13 $op rhs.13,
                    self.14 $op rhs.14,
                    self.15 $op rhs.15,
                )
            }
        }
    };

    (assn $trait:ident, $fn:ident, $typ:ty, $op:tt) => {
        impl $trait for $typ {
            fn $fn(&mut self, rhs: Self) {
                    self.0 $op rhs.0;
                    self.1 $op rhs.1;
                    self.2 $op rhs.2;
                    self.3 $op rhs.3;
                    self.4 $op rhs.4;
                    self.5 $op rhs.5;
                    self.6 $op rhs.6;
                    self.7 $op rhs.7;
                    self.8 $op rhs.8;
                    self.9 $op rhs.9;
                    self.10 $op rhs.10;
                    self.11 $op rhs.11;
                    self.12 $op rhs.12;
                    self.13 $op rhs.13;
                    self.14 $op rhs.14;
                    self.15 $op rhs.15;
            }
        }
    };
}

/// Implement operations for 32-lane SIMD types
macro_rules! impl_op32 {
    ($trait:ident, $fn:ident, $typ:ty, $op:tt) => {
        impl $trait for $typ {
            type Output = $typ;

            fn $fn(self, rhs: Self) -> Self::Output {
                Self(
                    self.0 $op rhs.0,
                    self.1 $op rhs.1,
                    self.2 $op rhs.2,
                    self.3 $op rhs.3,
                    self.4 $op rhs.4,
                    self.5 $op rhs.5,
                    self.6 $op rhs.6,
                    self.7 $op rhs.7,
                    self.8 $op rhs.8,
                    self.9 $op rhs.9,
                    self.10 $op rhs.10,
                    self.11 $op rhs.11,
                    self.12 $op rhs.12,
                    self.13 $op rhs.13,
                    self.14 $op rhs.14,
                    self.15 $op rhs.15,
                    self.16 $op rhs.16,
                    self.17 $op rhs.17,
                    self.18 $op rhs.18,
                    self.19 $op rhs.19,
                    self.20 $op rhs.20,
                    self.21 $op rhs.21,
                    self.22 $op rhs.22,
                    self.23 $op rhs.23,
                    self.24 $op rhs.24,
                    self.25 $op rhs.25,
                    self.26 $op rhs.26,
                    self.27 $op rhs.27,
                    self.28 $op rhs.28,
                    self.29 $op rhs.29,
                    self.30 $op rhs.30,
                    self.31 $op rhs.31,
                )
            }
        }
    };

    (assn $trait:ident, $fn:ident, $typ:ty, $op:tt) => {
        impl $trait for $typ {
            fn $fn(&mut self, rhs: Self) {
                self.0 $op rhs.0;
                self.1 $op rhs.1;
                self.2 $op rhs.2;
                self.3 $op rhs.3;
                self.4 $op rhs.4;
                self.5 $op rhs.5;
                self.6 $op rhs.6;
                self.7 $op rhs.7;
                self.8 $op rhs.8;
                self.9 $op rhs.9;
                self.10 $op rhs.10;
                self.11 $op rhs.11;
                self.12 $op rhs.12;
                self.13 $op rhs.13;
                self.14 $op rhs.14;
                self.15 $op rhs.15;
                self.16 $op rhs.16;
                self.17 $op rhs.17;
                self.18 $op rhs.18;
                self.19 $op rhs.19;
                self.20 $op rhs.20;
                self.21 $op rhs.21;
                self.22 $op rhs.22;
                self.23 $op rhs.23;
                self.24 $op rhs.24;
                self.25 $op rhs.25;
                self.26 $op rhs.26;
                self.27 $op rhs.27;
                self.28 $op rhs.28;
                self.29 $op rhs.29;
                self.30 $op rhs.30;
                self.31 $op rhs.31;
            }
        }
    };
}
