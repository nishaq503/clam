//! The individual algorithms that make up the CHAODA ensemble.

use distances::Number;

use crate::utils;

use super::{Graph, OddBall};

mod cc;
mod gn;
mod sc;
mod vd;

#[allow(unused_imports)]
pub use cc::CC;
#[allow(unused_imports)]
pub use gn::GN;
#[allow(unused_imports)]
pub use sc::SC;
#[allow(unused_imports)]
pub use vd::VD;

/// A trait for an algorithm in the CHAODA ensemble.
pub trait Algorithm<U: Number, C: OddBall<U, N>, const N: usize> {
    /// Evaluate the algorithm on a `Graph` and return a vector of scores for each
    /// `OddBall` in the `Graph`.
    ///
    /// The output vector must be the same length as the number of `OddBall`s in
    /// the `Graph`, and the order of the scores must correspond to the order of the
    /// `OddBall`s in the `Graph`.
    fn evaluate(&self, g: &mut Graph<U, C, N>) -> Vec<f32>;

    /// Whether to normalize anomaly scores by cluster or by point.
    fn normalize_by_cluster(&self) -> bool;

    /// Have points inherit scores from `OddBall`s.
    fn inherit_scores(&self, g: &Graph<U, C, N>, scores: &[f32]) -> Vec<f32> {
        let mut points_scores = vec![0.0; g.population()];
        for (c, &s) in g.iter_clusters().zip(scores.iter()) {
            for i in c.indices() {
                points_scores[i] = s;
            }
        }
        points_scores
    }

    /// Compute the anomaly scores for all points in the `Graph`.
    ///
    /// This method is a convenience method that wraps the `evaluate` and `inherit_scores`
    /// methods. It evaluates the algorithm on the `Graph` and then inherits the scores
    /// from the `OddBall`s to the points. It correctly handles normalization by cluster
    /// or by point.
    ///
    /// # Returns
    ///
    /// * A vector of anomaly scores for each point in the `Graph`.
    fn call(&self, g: &mut Graph<U, C, N>) -> Vec<f32> {
        let cluster_scores = {
            let scores = self.evaluate(g);
            if self.normalize_by_cluster() {
                let mean = utils::mean(&scores);
                let sd = utils::standard_deviation(&scores);
                utils::normalize_1d(&scores, mean, sd)
            } else {
                scores
            }
        };

        let scores = self.inherit_scores(g, &cluster_scores);
        if self.normalize_by_cluster() {
            scores
        } else {
            let mean = utils::mean(&scores);
            let sd = utils::standard_deviation(&scores);
            utils::normalize_1d(&scores, mean, sd)
        }
    }
}
