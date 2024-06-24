//! Relative Parent Cardinality algorithm.

use std::collections::HashMap;

use distances::Number;

use crate::chaoda::{Graph, OddBall};

use super::Algorithm;

/// `Cluster`s with a smaller fraction of points from their parent `Cluster` are more anomalous.
pub struct ParentCardinality;

impl<U: Number> Algorithm<U> for ParentCardinality {
    fn evaluate(&self, g: &mut Graph<U>) -> Vec<f32> {
        g.accumulated_cp_car_ratios()
    }

    fn normalize_by_cluster(&self) -> bool {
        true
    }
}
