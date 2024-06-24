//! Vertex Degree Algorithm

use distances::Number;

use crate::chaoda::{Graph, OddBall};

use super::Algorithm;

/// `Cluster`s with relatively few neighbors are more likely to be anomalous.
pub struct VertexDegree;

impl<U: Number> Algorithm<U> for VertexDegree {
    fn evaluate(&self, g: &mut Graph<U>) -> Vec<f32> {
        g.iter_neighbors().map(|n| -n.len().as_f32()).collect()
    }

    fn normalize_by_cluster(&self) -> bool {
        true
    }
}
