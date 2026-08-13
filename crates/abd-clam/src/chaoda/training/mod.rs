//! Functions and traits for training meta-ML prediction algorithms for ranking `Cluster`s before creating `Graph`s.

#![expect(dead_code, unused_imports, unused_variables, unused_mut, unreachable_code)]

use ndarray::prelude::*;

use crate::{DistanceValue, PartitionStrategy, Tree};

use super::meta_ml::{Layer, MetaMlPredictor, MetaMlTrainer};

mod gen_samples;

pub use gen_samples::{gen_training_sample_single, gen_training_samples_chaoda, gen_training_samples_graphs};

/// Trains Meta-ML models.
pub fn train_models<I, T, Metric, MetaMlAlg>(items: &[(bool, I)], metrics: Vec<Metric>) -> Result<Vec<MetaMlAlg>, String>
where
    T: DistanceValue,
    Metric: AsRef<dyn Fn(&I, &I) -> T>,
    MetaMlAlg: AsRef<dyn MetaMlTrainer<T, ()>>,
{
    let items_by_ref = items.iter().map(|(b, item)| (*b, item)).collect::<Vec<_>>();

    let trees = metrics
        .into_iter()
        .map(|metric| Box::new(move |a: &&I, b: &&I| (metric.as_ref())(a, b)))
        .map(|metric| {
            Tree::new(items_by_ref.clone(), metric, &|_| (), &|c: &_| c.cardinality > 2, &PartitionStrategy::default()).map(Tree::annotate_anomaly_features)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let max_depth = trees.iter().map(Tree::max_depth).max().unwrap_or(0);
    let initial_models = layer_models(max_depth);

    let max_epochs = 10; // TODO(Najib): Tune this based on convergence of the training process.
    let mut epoch = 0;
    let mut trained_models = Vec::new();
    loop {
        todo!();

        epoch += 1;
        if epoch > max_epochs {
            break;
        }
    }

    Ok(trained_models)
}

/// Generates a set of `MetaMlModel::Layer` models for each depth up to the maximum depth of the trees in the training data.
fn layer_models(max_depth: usize) -> Vec<Layer> {
    // TODO(Najib): Tune the step size based on the distribution of tree depths in the training data.
    let step_size = 5;
    (0..=max_depth).step_by(step_size).map(Layer::new).collect()
}
