//! The physics for the mass-spring system.

mod mass;
mod spring;
mod system;

use std::collections::HashMap;

pub use mass::Mass;
pub use spring::Spring;
pub use system::System;

/// A `HashMap` of `Mass`es, keyed by their `(offset, cardinality)`.
pub type MassMap<const DIM: usize> = HashMap<(usize, usize), Mass<DIM>>;

/// A `HashMap` of `Spring`s, keyed by the pairs of keys of the `Mass`es
pub type SpringMap<U, const DIM: usize> = HashMap<((usize, usize), (usize, usize)), Spring<U, DIM>>;
