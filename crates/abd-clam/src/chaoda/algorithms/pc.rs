//! Relative Parent Cardinality algorithm.

use std::collections::HashMap;

use distances::Number;

use crate::chaoda::{Graph, OddBall};

use super::Algorithm;

/// `Cluster`s with a smaller fraction of points from their parent `Cluster` are more anomalous.
pub struct ParentCardinality;

impl<U: Number, C: OddBall<U, N>, const N: usize> Algorithm<U, C, N> for ParentCardinality {
    fn evaluate(&self, g: &mut Graph<U, C, N>) -> Vec<f32> {
        g.iter_clusters().map(C::accumulated_cp_car_ratio).collect()
    }

    fn normalize_by_cluster(&self) -> bool {
        true
    }
}
