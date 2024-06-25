//! Multi-Tree Chaoda

use distances::Number;
use rayon::prelude::*;

use crate::{
    chaoda::{Member, MlModel, OddBall},
    Dataset, Instance, PartitionCriterion,
};

use super::st_chaoda::SingleTreeChaoda;

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
    #[allow(clippy::type_complexity)]
    pub fn new<P: PartitionCriterion<U>>(
        data: &D,
        criteria: &P,
        seed: Option<u64>,
        metrics_expense_names: Vec<(fn(&I, &I) -> U, bool, String)>,
    ) -> Self {
        let trees = metrics_expense_names
            .into_iter()
            .map(|(metric, is_expensive, name)| {
                let data = data.clone_with_new_metric(metric, is_expensive, name);
                SingleTreeChaoda::new(data, criteria, seed)
            })
            .collect();
        Self { trees }
    }

    /// Get each dataset in the ensemble.
    #[must_use]
    pub fn datasets(&self) -> Vec<&D> {
        self.trees.iter().map(SingleTreeChaoda::data).collect()
    }

    /// Get each root cluster in the ensemble.
    #[must_use]
    pub fn roots(&self) -> Vec<&C> {
        self.trees.iter().map(SingleTreeChaoda::root).collect()
    }

    /// Train the ensemble.
    ///
    /// # Parameters
    ///
    /// * `num_epochs`: The number of epochs to train the ensemble.
    /// * `labels`: The labels for the data. `true` indicates an anomaly.
    /// * `min_depth`: The minimum depth in the tree at which to consider a cluster for selection.
    ///
    /// # Errors
    ///
    /// * If the number of labels does not match the number of instances.
    /// * if the number of meta-ML models is not a multiple of the number of algorithms.
    ///
    /// # Returns
    ///
    /// The pairs of (meta-ml model, name of chaoda algorithm) used in the ensemble.
    pub fn train(
        &mut self,
        num_epochs: usize,
        labels: &[bool],
        min_depth: usize,
    ) -> Result<Vec<(String, MlModel)>, String> {
        Ok(self
            .trees
            .par_iter_mut()
            .map(|tree| {
                let algorithms = Member::default_members()
                    .into_iter()
                    .map(|member| (member, MlModel::defaults()))
                    .collect();
                tree.train(num_epochs, labels, min_depth, algorithms)
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect())
    }
}
