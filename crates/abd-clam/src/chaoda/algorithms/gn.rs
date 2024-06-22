//! Graph Neighborhood Algorithm

use distances::Number;

use crate::chaoda::{Graph, OddBall};

use super::Algorithm;

/// `Cluster`s in an isolated neighborhood are more likely to be anomalous.
pub struct GN {
    /// The fraction of graph diameter to use as the neighborhood radius.
    diameter_fraction: f32,
}

impl GN {
    /// Create a new `GN` algorithm.
    ///
    /// # Parameters
    ///
    /// * `diameter_fraction`: The fraction of graph diameter to use as the neighborhood radius.
    #[allow(dead_code)]
    pub fn new(diameter_fraction: f32) -> Result<Self, String> {
        if diameter_fraction <= 0.0 || diameter_fraction >= 1.0 {
            Err("Diameter fraction must be in the range [0, 1]".to_string())
        } else {
            Ok(Self { diameter_fraction })
        }
    }
}

impl<U: Number, C: OddBall<U, N>, const N: usize> Algorithm<U, C, N> for GN {
    fn evaluate(&self, g: &mut Graph<U, C, N>) -> Vec<f32> {
        let diameter = g.diameter();
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let k = (self.diameter_fraction * diameter.as_f32()).round() as usize;
        g.neighborhood_sizes()
            .iter()
            .map(|&n| if n.len() <= k { n.last().unwrap_or(&0) } else { &n[k] })
            .map(|&n| -n.as_f32())
            .collect()
    }

    fn normalize_by_cluster(&self) -> bool {
        true
    }
}
