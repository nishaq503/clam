//! Redesign of CLAM
// #![doc = include_str!("../README.md")]

pub mod cakes;
mod core;
// mod utils;

pub use core::{Ball, DistanceValue, FloatDistanceValue};

// The utils module is for internal use only.
pub use core::utils;

// pub use utils::sz_lev_builder;

/// The current version of the crate.
pub const VERSION: &str = "0.32.0";
