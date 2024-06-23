//! Clustered Hierarchical Anomaly and Outlier Detection Algorithms (CHAODA)

mod algorithms;
mod cluster;
mod graph;
mod meta_ml;
mod stc;

pub use algorithms::Algorithm;
pub use cluster::{OddBall, Ratios, Vertex};
use distances::Number;
pub use graph::Graph;
pub use meta_ml::Model;
use stc::SingleTreeChaoda;

use crate::{Dataset, Instance, PartitionCriterion};

/// A CHAODA ensemble.
///
/// # Type Parameters
///
/// * `I`: The type of the instances in the dataset.
/// * `U`: The type of the distance values in the dataset.
/// * `D`: The type of the dataset.
/// * `C`: The type of the `OddBall` in the ensemble.
/// * `N`: The number of anomaly ratios in the `OddBall`.
pub struct Chaoda<I, U, D, C, const N: usize>
where
    I: Instance,
    U: Number,
    D: Dataset<I, U> + Clone,
    C: OddBall<U, N>,
{
    /// The Chaoda sub-ensembles on each tree.
    trees: Vec<SingleTreeChaoda<I, U, D, C, N>>,
}

impl<I, U, D, C, const N: usize> Chaoda<I, U, D, C, N>
where
    I: Instance,
    U: Number,
    D: Dataset<I, U> + Clone,
    C: OddBall<U, N>,
{
    /// Create a new `Chaoda` ensemble.
    pub fn new<P: PartitionCriterion<U>>(data: &D, criteria: &P, seed: Option<u64>, num_trees: usize) -> Self {
        let mut trees = Vec::with_capacity(num_trees);
        for _ in 0..num_trees {
            trees.push(SingleTreeChaoda::new(data.clone(), criteria, seed, None));
        }
        Self { trees }
    }

    /// Get each dataset in the ensemble.
    #[must_use]
    pub fn datasets(&self) -> Vec<&D> {
        self.trees.iter().map(stc::SingleTreeChaoda::data).collect()
    }

    /// Get each root cluster in the ensemble.
    #[must_use]
    pub fn roots(&self) -> Vec<&C> {
        self.trees.iter().map(stc::SingleTreeChaoda::root).collect()
    }

    /// Get each set of algorithms in the ensemble.
    #[must_use]
    pub fn algorithms(&self) -> Vec<&[Box<dyn Algorithm<U, C, N>>]> {
        self.trees.iter().map(stc::SingleTreeChaoda::algorithms).collect()
    }
}
