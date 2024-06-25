//! Relative Parent Cardinality algorithm.

use distances::Number;

use crate::chaoda::Graph;

use super::Algorithm;

/// `Cluster`s with a smaller fraction of points from their parent `Cluster` are more anomalous.
#[derive(Clone)]
pub struct ParentCardinality;

impl Algorithm for ParentCardinality {
    fn name(&self) -> String {
        "pc".to_string()
    }

    fn evaluate_clusters<U: Number, const N: usize>(&self, g: &mut Graph<U, N>) -> Vec<f32> {
        g.accumulated_cp_car_ratios()
    }

    fn normalize_by_cluster(&self) -> bool {
        true
    }
}

impl Default for ParentCardinality {
    fn default() -> Self {
        Self
    }
}
