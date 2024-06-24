//! Subgraph Cardinality algorithm.

use distances::Number;

use crate::chaoda::{Graph, OddBall};

use super::Algorithm;

/// `Cluster`s in subgraphs with relatively small population are more likely to be anomalous.
pub struct SubgraphCardinality;

impl<U: Number> Algorithm<U> for SubgraphCardinality {
    fn evaluate(&self, g: &mut Graph<U>) -> Vec<f32> {
        g.iter_components()
            .flat_map(|sg| {
                let p = -sg.population().as_f32();
                core::iter::repeat(p).take(sg.cardinality())
            })
            .collect()
    }

    fn normalize_by_cluster(&self) -> bool {
        true
    }
}
