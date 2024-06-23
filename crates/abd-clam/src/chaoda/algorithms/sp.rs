//! Stationary Probabilities Algorithm.

use distances::Number;

use crate::chaoda::{Graph, OddBall};

use super::Algorithm;

/// Clusters with smaller stationary probabilities are more anomalous.
pub struct StationaryProbability {
    /// The Random Walk will be simulated for 2^`num_steps` steps.
    num_steps: usize,
}

impl StationaryProbability {
    /// Create a new `StationaryProbability` algorithm.
    ///
    /// # Arguments
    ///
    /// * `num_steps`: The Random Walk will be simulated for 2^`num_steps` steps.
    pub const fn new(num_steps: usize) -> Self {
        Self { num_steps }
    }
}

impl<U: Number, C: OddBall<U, N>, const N: usize> Algorithm<U, C, N> for StationaryProbability {
    fn evaluate(&self, g: &mut Graph<U, C, N>) -> Vec<f32> {
        g.compute_stationary_probabilities(self.num_steps)
    }

    fn normalize_by_cluster(&self) -> bool {
        true
    }
}
