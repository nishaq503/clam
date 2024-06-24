//! Cluster Cardinality algorithm.

use distances::Number;

use crate::chaoda::{Graph, OddBall};

use super::Algorithm;

/// `Cluster`s with relatively few points are more likely to be anomalous.
pub struct ClusterCardinality;

impl<U: Number> Algorithm<U> for ClusterCardinality {
    fn evaluate(&self, g: &mut Graph<U>) -> Vec<f32> {
        g.iter_clusters().map(|&(_, c)| -c.as_f32()).collect()
    }

    fn normalize_by_cluster(&self) -> bool {
        true
    }
}
