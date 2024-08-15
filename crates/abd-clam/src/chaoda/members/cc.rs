//! Cluster Cardinality algorithm.

use distances::Number;
use serde::{Deserialize, Serialize};

use crate::{chaoda::Graph, Cluster, Dataset};

use super::Algorithm;

/// `Cluster`s with relatively few points are more likely to be anomalous.
#[derive(Clone, Serialize, Deserialize)]
pub struct ClusterCardinality;

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> Algorithm<I, U, D, S> for ClusterCardinality {
    fn name(&self) -> String {
        "cc".to_string()
    }

    fn evaluate_clusters(&self, g: &mut Graph<I, U, D, S>) -> Vec<f32> {
        g.iter_clusters().map(|c| -c.cardinality().as_f32()).collect()
    }

    fn normalize_by_cluster(&self) -> bool {
        true
    }
}

impl Default for ClusterCardinality {
    fn default() -> Self {
        Self
    }
}
