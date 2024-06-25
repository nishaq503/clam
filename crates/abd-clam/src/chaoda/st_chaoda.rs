//! Chaoda on a single tree.

use distances::Number;
use ndarray::prelude::*;
use smartcore::{linalg::basic::matrix::DenseMatrix, metrics::roc_auc_score};

use crate::{Dataset, Instance, PartitionCriterion};

use super::{Graph, Member, MlModel, OddBall};

/// A CHAODA ensemble on a single tree.
///
/// # Type Parameters
///
/// * `I`: The type of the instances in the dataset.
/// * `U`: The type of the distance values in the dataset.
/// * `D`: The type of the dataset.
/// * `C`: The type of the `OddBall` in the ensemble.
/// * `N`: The number of anomaly ratios in the `OddBall`.
pub struct SingleTreeChaoda<I, U, D, C, const N: usize>
where
    I: Instance,
    U: Number,
    D: Dataset<I, U>,
    C: OddBall<U, N>,
{
    /// The data.
    data: D,
    /// The root `Cluster` of the tree.
    root: C,
    /// The pairs of Chaoda member and associated `Graph`s for each member.
    member_graphs: Vec<(Member, Vec<Graph<U, N>>)>,
    /// Phantom data to satisfy the compiler.
    _phantom: std::marker::PhantomData<I>,
}

impl<I, U, D, C, const N: usize> SingleTreeChaoda<I, U, D, C, N>
where
    I: Instance,
    U: Number,
    D: Dataset<I, U>,
    C: OddBall<U, N>,
{
    /// Create a new `SingleTreeChaoda` ensemble.
    pub fn new<P: PartitionCriterion<U>>(mut data: D, criteria: &P, seed: Option<u64>) -> Self {
        let root = C::new_root(&data, seed).partition(&mut data, criteria, seed);
        Self {
            data,
            root,
            _phantom: std::marker::PhantomData,
            member_graphs: Vec::new(),
        }
    }

    /// Train the ensemble.
    ///
    /// # Parameters
    ///
    /// * `num_epochs`: The number of epochs to train the ensemble.
    /// * `labels`: The labels for the data. `true` indicates an anomaly.
    /// * `min_depth`: The minimum depth in the tree at which to consider a cluster for selection.
    /// * `algorithms`: The pairs of `CHAODA` algorithm and collection of `meta_ml` models to use in the ensemble.
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
        mut algorithms: Vec<(Member, Vec<MlModel>)>,
    ) -> Result<Vec<(String, MlModel)>, String> {
        // Check that the number of labels matches the number of instances.
        if labels.len() != self.data.cardinality() {
            return Err("The number of labels must match the number of instances".to_string());
        }

        let labels = labels.iter().map(|&l| if l { 1.0 } else { 0.0 }).collect::<Vec<f32>>();

        let scorer = |clusters: &[&C]| {
            clusters
                .iter()
                .map(|c| {
                    if c.depth() == min_depth || (c.is_leaf() && c.depth() < min_depth) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>()
        };
        let graph = Graph::from_tree(&self.root, &self.data, scorer, 4);
        let mut graphs = algorithms
            .iter()
            .map(|(_, models)| models.iter().map(|_| graph.clone()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let mut full_train_x = algorithms
            .iter()
            .map(|(_, models)| models.iter().map(|_| Vec::<Vec<f32>>::new()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let mut full_train_y = algorithms
            .iter()
            .map(|(_, models)| models.iter().map(|_| Vec::<f32>::new()).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        // Train the ensemble.
        for _ in 0..num_epochs {
            for ((((alg, meta_ml_models), alg_graphs), alg_train_x), alg_train_y) in algorithms
                .iter_mut()
                .zip(graphs.iter_mut())
                .zip(full_train_x.iter_mut())
                .zip(full_train_y.iter_mut())
            {
                // Run the CHAODA algorithms on each graph and create training data
                for (((ml_model, graph), train_x), train_y) in meta_ml_models
                    .iter_mut()
                    .zip(alg_graphs.iter_mut())
                    .zip(alg_train_x.iter_mut())
                    .zip(alg_train_y.iter_mut())
                {
                    // For each cluster, get the anomaly properties
                    train_x.extend(graph.iter_anomaly_properties().map(|(p, p_)| {
                        let mut properties = p.to_vec();
                        properties.extend_from_slice(p_);
                        properties
                    }));

                    // Evaluate the algorithm on the graph to get anomaly for clusters
                    let anomaly_ratings = Member::normalize_scores(&alg.evaluate_clusters(graph));
                    // For each cluster, get a roc-auc score
                    train_y.extend(graph.iter_clusters().zip(anomaly_ratings.iter()).map(
                        |(&(start, cardinality), &rating)| {
                            let mut y_true = labels[start..(start + cardinality)].to_vec();
                            y_true.push(1.0);
                            y_true.push(0.0);
                            let mut y_pred = vec![rating; cardinality];
                            y_pred.push(1.0);
                            y_pred.push(0.0);
                            roc_auc_score(&y_true, &y_pred).as_f32()
                        },
                    ));

                    // Train the meta-ML model with the updated data
                    let data = DenseMatrix::from_2d_vec(train_x);
                    let target = Array1::from_vec(train_y.clone());
                    ml_model.train(&data, &target)?;

                    // Update graphs
                    let cluster_scorer = |clusters: &[&C]| {
                        let anomaly_properties = clusters
                            .iter()
                            .map(|c| {
                                let (p, p_) = c.ratios();
                                let mut properties = p.to_vec();
                                properties.extend_from_slice(p_.as_ref());
                                properties
                            })
                            .collect::<Vec<_>>();
                        let properties = DenseMatrix::from_2d_vec(&anomaly_properties);
                        ml_model
                            .predict(&properties)
                            .unwrap_or_else(|_| unreachable!("We made sure the shape was correct."))
                            .to_vec()
                    };
                    *graph = Graph::from_tree(&self.root, &self.data, cluster_scorer, min_depth);
                }
            }
        }

        self.member_graphs = algorithms
            .iter()
            .map(|(member, _)| member.clone())
            .zip(graphs)
            .collect();

        let out_names = algorithms
            .into_iter()
            .flat_map(|(member, models)| models.into_iter().map(move |model| (member.name(), model)))
            .collect::<Vec<_>>();

        Ok(out_names)
    }

    /// Get the root `Cluster` of the tree.
    pub const fn root(&self) -> &C {
        &self.root
    }

    /// Get the data.
    pub const fn data(&self) -> &D {
        &self.data
    }
}
