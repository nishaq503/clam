//! Clustered Hierarchical Anomaly and Outlier Detection Algorithms (CHAODA)

mod cluster;
mod graph;
mod members;
mod meta_ml;

pub use cluster::{OddBall, Ratios, Vertex};
pub use graph::Graph;
pub use members::Member;
pub use meta_ml::MlModel;

use distances::Number;
use ndarray::prelude::*;
use rayon::prelude::*;
use smartcore::{linalg::basic::matrix::DenseMatrix, metrics::roc_auc_score};

use crate::{Dataset, Instance};

/// The training data for the ensemble.
///
/// The outer vector is the ensemble members.
/// The middle vector is the meta-ML models.
/// The inner vector is the epoch training data.
/// The tuple is the anomaly ratios and roc scores.
pub type TrainingData = Vec<Vec<(Vec<Vec<f32>>, Vec<f32>)>>;

/// A CHAODA ensemble.
pub struct Chaoda {
    /// The combination of the CHAODA algorithms and the meta-ML models.
    algorithms: Vec<(Member, Vec<MlModel>)>,
    /// The minimum depth of `Cluster`s to consider for selection.
    min_depth: usize,
}

impl Default for Chaoda {
    fn default() -> Self {
        Self {
            algorithms: Member::default_members()
                .into_iter()
                .map(|member| (member, MlModel::defaults()))
                .collect(),
            min_depth: 4,
        }
    }
}

impl Chaoda {
    /// Create a new `Chaoda` ensemble.
    #[must_use]
    pub const fn new(algorithms: Vec<(Member, Vec<MlModel>)>, min_depth: usize) -> Self {
        Self { algorithms, min_depth }
    }

    /// Predict the anomaly scores for the given dataset and root `Cluster`.
    pub fn predict<I, U, D, C, const N: usize>(&self, data: &D, root: &C) -> Vec<f32>
    where
        I: Instance,
        U: Number,
        D: Dataset<I, U>,
        C: OddBall<U, N>,
    {
        let mut graphs = self.create_graphs(data, root);
        let predictions = self
            .algorithms
            .par_iter()
            .zip(graphs.par_iter_mut())
            .flat_map(|((member, _), m_graphs)| m_graphs.par_iter_mut().map(|g| member.evaluate_points(g)))
            .collect::<Vec<_>>();

        let num_predictions = predictions.len();
        let predictions = predictions.into_iter().flatten().collect::<Vec<_>>();
        let predictions = Array2::from_shape_vec((num_predictions, data.cardinality()), predictions)
            .unwrap_or_else(|_| unreachable!("We made sure the shape was correct."));

        // Take the mean of the predictions for each point.
        predictions
            .mean_axis(Axis(0))
            .unwrap_or_else(|| unreachable!("We made sure no axis was empty."))
            .to_vec()
    }

    /// Train the ensemble on the given dataset.
    pub fn train<I, U, D, C, const N: usize>(
        &mut self,
        data: &D,
        root: &C,
        labels: &[bool],
        num_epochs: usize,
        previous_data: Option<TrainingData>,
    ) -> TrainingData
    where
        I: Instance,
        U: Number,
        D: Dataset<I, U>,
        C: OddBall<U, N>,
    {
        let mut graphs = if previous_data.is_some() {
            self.create_graphs(data, root)
        } else {
            let cluster_scorer = |clusters: &[&C]| {
                clusters
                    .iter()
                    .map(|c| {
                        if c.depth() == self.min_depth || (c.is_leaf() && c.depth() < self.min_depth) {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<_>>()
            };
            let graph = Graph::from_tree(root, data, cluster_scorer, 4);
            self.algorithms
                .par_iter()
                .map(|(_, models)| models.par_iter().map(|_| graph.clone()).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        };

        let labels = labels.iter().map(|&l| if l { 1.0 } else { 0.0 }).collect::<Vec<f32>>();
        let mut full_training_data = previous_data.unwrap_or_default();

        for _ in 0..num_epochs {
            let new_training_data = self.generate_training_data(&mut graphs, &labels);
            full_training_data
                .par_iter_mut()
                .zip(new_training_data)
                .for_each(|(m_old, m_new)| {
                    m_old
                        .par_iter_mut()
                        .zip(m_new)
                        .for_each(|((x_old, y_old), (x_new, y_new))| {
                            x_old.extend(x_new);
                            y_old.extend(y_new);
                        });
                });

            self.train_inner_models(&full_training_data);
            graphs = self.create_graphs(data, root);
        }

        full_training_data
    }

    /// Create `Graph`s for the ensemble.
    fn create_graphs<I, U, D, C, const N: usize>(&self, data: &D, root: &C) -> Vec<Vec<Graph<U, N>>>
    where
        I: Instance,
        U: Number,
        D: Dataset<I, U>,
        C: OddBall<U, N>,
    {
        self.algorithms
            .par_iter()
            .map(|(_, models)| {
                models
                    .par_iter()
                    .map(|ml_model| {
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
                        Graph::from_tree(root, data, cluster_scorer, self.min_depth)
                    })
                    .collect()
            })
            .collect()
    }

    /// Generate training data from `Graph`s.
    fn generate_training_data<U: Number, const N: usize>(
        &self,
        graphs: &mut [Vec<Graph<U, N>>],
        labels: &[f32],
    ) -> TrainingData {
        self.algorithms
            .par_iter()
            .zip(graphs)
            .map(|((member, _), m_graphs)| {
                m_graphs
                    .par_iter_mut()
                    .map(|g| {
                        let train_x = g
                            .iter_anomaly_properties()
                            .map(|(p, p_)| {
                                let mut properties = p.to_vec();
                                properties.extend_from_slice(p_.as_ref());
                                properties
                            })
                            .collect::<Vec<_>>();
                        let anomaly_ratings = Member::normalize_scores(&member.evaluate_clusters(g));
                        let train_y = g
                            .iter_clusters()
                            .zip(anomaly_ratings)
                            .map(|(&(start, cardinality), rating)| {
                                let mut y_true = labels[start..(start + cardinality)].to_vec();
                                y_true.push(1.0);
                                y_true.push(0.0);
                                let mut y_pred = vec![rating; cardinality];
                                y_pred.push(1.0);
                                y_pred.push(0.0);
                                roc_auc_score(&y_true, &y_pred).as_f32()
                            })
                            .collect::<Vec<f32>>();
                        (train_x, train_y)
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Train the inner models given the training data.
    fn train_inner_models(&mut self, training_data: &TrainingData) {
        self.algorithms
            .par_iter_mut()
            .zip(training_data)
            .for_each(|((_, ml_models), m_data)| {
                ml_models
                    .par_iter_mut()
                    .zip(m_data)
                    .for_each(|(model, (train_x, train_y))| {
                        let train_x = DenseMatrix::from_2d_vec(train_x);
                        let train_y = Array1::from_vec(train_y.clone());
                        model
                            .train(&train_x, &train_y)
                            .unwrap_or_else(|_| unreachable!("We made sure the shape was correct."));
                    });
            });
    }
}
