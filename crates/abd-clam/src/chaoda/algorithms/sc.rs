//! Subgraph Cardinality algorithm.

use distances::Number;

use crate::chaoda::{Graph, OddBall};

use super::Algorithm;

/// `Cluster`s in subgraphs with relatively small population are more likely to be anomalous.
pub struct SC;

impl<U: Number, C: OddBall<U, N>, const N: usize> Algorithm<U, C, N> for SC {
    fn evaluate(&self, g: &mut Graph<U, C, N>) -> Vec<f32> {
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
