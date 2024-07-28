//! Traits to adapt a `Ball` into other `Cluster` types.

use distances::Number;

use super::{Ball, Cluster};

/// A trait for the parameters to use for adapting a `Ball` into another `Cluster`.
pub trait Params<U: Number>: Default {
    /// Given the `Ball` that was adapted into a `Cluster`, returns parameters
    /// to use for adapting the children of the `Ball`.
    #[must_use]
    fn child_params<B: AsRef<Ball<U>>>(&self, child_balls: &[B]) -> Vec<Self>;
}

/// A trait for adapting a `Ball` into another `Cluster` type.
///
/// # Parameters
///
/// - `U`: The type of the distance values.
/// - `P`: The type of the parameters to use for the adaptation.
pub trait Adapter<U: Number, P: Params<U>>: Cluster<U> {
    /// Adapts a tree of `Ball`s into a `Cluster`.
    fn adapt(ball: Ball<U>, params: Option<P>) -> (Self, Vec<usize>)
    where
        Self: Sized;

    /// Returns the `Ball` that was adapted into this `Cluster`. This should not
    /// have any children.
    fn ball(&self) -> &Ball<U>;

    /// Returns the `Ball` mutably that was adapted into this `Cluster`. This
    /// should not have any children.
    fn ball_mut(&mut self) -> &mut Ball<U>;
}

/// Parallel version of the `Params` trait.
pub trait ParParams<U: Number>: Params<U> + Send + Sync {
    /// Parallel version of the `child_params` method.
    #[must_use]
    fn par_child_params<B: AsRef<Ball<U>>>(&self, child_balls: &[B]) -> Vec<Self>;
}

/// Parallel version of the `Adapter` trait.
#[allow(clippy::module_name_repetitions)]
pub trait ParAdapter<U: Number, P: ParParams<U>>: Adapter<U, P> + Send + Sync {
    /// Parallel version of the `adapt` method.
    fn par_adapt(ball: Ball<U>, params: Option<P>) -> (Self, Vec<usize>)
    where
        Self: Sized;
}