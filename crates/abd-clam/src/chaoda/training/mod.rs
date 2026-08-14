//! Functions and traits for training meta-ML prediction algorithms for ranking `Cluster`s before creating `Graph`s.

use crate::{DistanceValue, Tree};

use super::{
    Graph, algorithms,
    meta_ml::{DecisionTree, Layer, LinearRegression, MetaMlPredictor},
};

mod gen_samples;

pub use gen_samples::gen_training_sample;

/// A type-alias for the nested vector of trained Meta-ML models.
///
/// - The outer vector corresponds to the algorithms.
/// - The inner vector corresponds to the meta-ml models.
pub type TrainedModels<T, A> = Vec<Vec<Box<dyn MetaMlPredictor<T, A>>>>;

/// Trains Meta-ML models for a single tree.
///
/// # Arguments
///
/// - `tree`: A pre-built tree to use for training the models. The tree will be annotated with anomaly features before training and de-annotated just before
///   returning from this function.
/// - `algorithms`: A list of the graph algorithms to use.
/// - `oracle`: A function that takes item IDs and returns whether the item is an anomaly or not. This is used to guide the training of the models.
/// - `initial_layer_depths`: The depths of the initial layers for the Meta-ML models: [`min_depth`, `max_depth`, `step_size`]. Training will be begin with
///   layer-graphs of each eligible depth of clusters from the tree.
///
/// # Returns
///
/// The restored tree and the trained meta-ml models as a nested vector: [`TrainedModels<T, A>`].
///
/// # Errors
///
/// - If the ROC scores fail to compute.
/// - Training any model fails.
#[expect(clippy::type_complexity)]
#[expect(unused_variables, unused_mut)]
pub fn train_models<Id, I, T, A, M, Alg, Oracle>(
    tree: Tree<Id, I, T, A, M>,
    algorithms: &[Alg],
    oracle: &Oracle,
    initial_layer_depths: [usize; 3],
) -> Result<(Tree<Id, I, T, A, M>, TrainedModels<T, A>), String>
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
    Alg: AsRef<dyn algorithms::GraphAlgorithm<Id, I, T, A, M>>,
    Oracle: Fn(&Id) -> bool,
{
    let chaoda_tree = tree.annotate_anomaly_features();
    let tree_depth = chaoda_tree.max_depth();

    let [min_depth, max_depth, step_size] = initial_layer_depths;
    let max_depth = max_depth.min(tree_depth);

    // Use layer graphs to bootstrap training.
    let layer_graphs = (min_depth..=max_depth)
        .step_by(step_size)
        .map(Layer::new)
        .map(|m| Box::new(m) as Box<dyn MetaMlPredictor<T, A>>)
        .map(|predictor| {
            let (directly_selected, ancestors) = chaoda_tree.select_chaoda_clusters(&predictor, min_depth);
            Graph::from_tree(&chaoda_tree, &directly_selected, &ancestors)
        })
        .collect::<Vec<_>>();

    // Generate the first batch of training data using the layer graphs.
    let mut training_data = algorithms
        .iter()
        .map(|alg| {
            layer_graphs
                .iter()
                .map(|graph| gen_training_sample(&chaoda_tree, graph, alg, oracle))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Initialize the meta-ml models for each graph algorithm. These models will be trained over the epochs.
    let mut models_in_training = algorithms
        .iter()
        .map(|_| {
            vec![
                Box::new(LinearRegression::new()) as Box<dyn MetaMlPredictor<T, A>>,
                Box::new(DecisionTree) as Box<dyn MetaMlPredictor<T, A>>,
            ]
        })
        .collect::<Vec<_>>();

    let max_epochs = 10; // TODO(Najib): Tune this based on convergence of the training process.
    for epoch in 0..max_epochs {
        for (alg, alg_models) in algorithms.iter().zip(models_in_training.iter_mut()) {
            for model in alg_models.iter_mut() {
                let (directly_selected, ancestors) = chaoda_tree.select_chaoda_clusters(model, min_depth);
                let graph = Graph::from_tree(&chaoda_tree, &directly_selected, &ancestors);
                todo!()
            }
        }
    }

    // Restore the original tree.
    let (tree, _) = chaoda_tree.decompound_annotations();
    Ok((tree, models_in_training))
}
