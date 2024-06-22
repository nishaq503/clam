//! Cluster Cardinality algorithm.

use distances::Number;

use crate::chaoda::{Graph, OddBall};

use super::Algorithm;

/// `Cluster`s with relatively few points are more likely to be anomalous.
pub struct CC;

impl<U: Number, C: OddBall<U, N>, const N: usize> Algorithm<U, C, N> for CC {
    fn evaluate(&self, g: &Graph<U, C, N>) -> Vec<f32> {
        g.iter_clusters().map(|c| -c.cardinality().as_f32()).collect()
    }

    fn normalize_by_cluster(&self) -> bool {
        true
    }
}
