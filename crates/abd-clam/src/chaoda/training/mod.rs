//! Functions and traits for training meta-ML prediction algorithms for ranking `Cluster`s before creating `Graph`s.

#![expect(dead_code, unused_imports, unused_variables, unused_mut)]

use ndarray::prelude::*;

use crate::{Cluster, DistanceValue, PartitionStrategy, Tree};

use super::{
    algorithms,
    meta_ml::{Layer, MetaMlPredictor, MetaMlTrainer},
};

mod gen_samples;

pub use gen_samples::{gen_training_sample_single, gen_training_samples_chaoda, gen_training_samples_graphs};

/// Trains Meta-ML models.
///
/// # Arguments
///
/// - `trees`: Pre-built trees to use for training the models. The trees will be annotated with anomaly features before training and de-annotated just before
///   returning from this function.
/// - `algorithms`: A list of the graph algorithms to use.
/// - `oracle`: A function that takes item IDs and returns whether the item is an anomaly or not. This is used to guide the training of the models.
/// - `initial_layer_depths`: The depths of the initial layers for the Meta-ML models: [`min_depth`, `max_depth`, `step_size`]. Training will be begin with
///   layer-graphs of each eligible depth of clusters from the trees.
pub fn train_models<Id, I, T, A, Metric, Alg, Oracle>(
    trees: &mut Vec<Tree<Id, I, T, A, Metric>>,
    algorithms: &[Alg],
    oracle: &Oracle,
    initial_layer_depths: [usize; 3],
) -> Vec<Box<dyn MetaMlPredictor<T, A>>>
where
    T: DistanceValue,
    Metric: AsRef<dyn Fn(&I, &I) -> T>,
    Alg: AsRef<dyn algorithms::GraphAlgorithm<bool, I, T, (), Metric>>,
    Oracle: Fn(&Id) -> bool,
{
    let chaoda_trees = trees.drain(..).map(Tree::annotate_anomaly_features).collect::<Vec<_>>();

    let [min_depth, max_depth, step_size] = initial_layer_depths;
    let mut current_predictors = (min_depth..=max_depth)
        .step_by(step_size)
        .map(Layer::new)
        .map(|m| Box::new(m) as Box<dyn MetaMlPredictor<T, A>>)
        .collect::<Vec<_>>();

    let max_epochs = 10; // TODO(Najib): Tune this based on convergence of the training process.
    for epoch in 0..max_epochs {
        // For every combination of predictor and tree, generate training samples and train the next generation of predictors.
        let mut next_predictors = Vec::new();
        for tree in &chaoda_trees {
            for predictor in &current_predictors {
                todo!()
            }
        }

        current_predictors = next_predictors;
    }

    // Restore the trees to the original vector.
    trees.extend(chaoda_trees.into_iter().map(|tree| tree.decompound_annotations().0));

    current_predictors
}
