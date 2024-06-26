//! Clustered Hierarchical Anomaly and Outlier Detection Algorithms (CHAODA)

mod cluster;
mod graph;
mod members;
mod meta_ml;

use std::path::Path;

pub use cluster::{OddBall, Ratios, Vertex};
pub use graph::Graph;
pub use members::Member;
pub use meta_ml::MlModel;

use distances::Number;
use ndarray::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use smartcore::metrics::roc_auc_score;

use crate::{Dataset, Instance};

/// The training data for the ensemble.
///
/// The outer vector is the ensemble members.
/// The middle vector is the meta-ML models.
/// The inner vector is the epoch training data.
/// The tuple is the anomaly ratios and roc scores.
pub type TrainingData = Vec<Vec<(Vec<Vec<f32>>, Vec<f32>)>>;

/// A CHAODA ensemble.
#[derive(Serialize, Deserialize)]
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
                .into_par_iter()
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

    /// Get the number of predictors in the ensemble.
    #[must_use]
    pub fn num_predictors(&self) -> usize {
        self.algorithms.iter().map(|(_, models)| models.len()).sum()
    }

    /// Save the model to a given path.
    ///
    /// # Arguments
    ///
    /// * `path`: The path to save the model to.
    ///
    /// # Errors
    ///
    /// * If there is an error creating the file.
    /// * If there is an error serializing the model.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let file = std::fs::File::create(path).map_err(|e| format!("Error creating file: {e}"))?;
        bincode::serialize_into(file, self).map_err(|e| format!("Error serializing: {e}"))?;
        Ok(())
    }

    /// Load the model from a given path.
    ///
    /// # Arguments
    ///
    /// * `path`: The path to load the model from.
    ///
    /// # Errors
    ///
    /// * If there is an error opening the file.
    /// * If there is an error deserializing the model.
    pub fn load(path: &Path) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("Error opening file: {e}"))?;
        let model = bincode::deserialize_from(file).map_err(|e| format!("Error deserializing: {e}"))?;
        Ok(model)
    }

    /// Predict the anomaly scores for the given dataset and root `Cluster`.
    pub fn predict<I, U, D, C, const N: usize>(&self, data: &D, root: &C) -> Array2<f32>
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

        let predictions = predictions.into_iter().flatten().collect::<Vec<_>>();
        Array2::from_shape_vec((self.num_predictors(), data.cardinality()), predictions)
            .unwrap_or_else(|_| unreachable!("We made sure the shape was correct."))
    }

    /// Aggregate the predictions of the ensemble.
    ///
    /// For now, we take the mean of the anomaly scores for each point. Later,
    /// we may want to consider other aggregation methods.
    #[must_use]
    pub fn aggregate_predictions(scores: &Array2<f32>) -> Array1<f32> {
        // Take the mean of the anomaly scores for each point
        scores
            .mean_axis(Axis(0))
            .unwrap_or_else(|| unreachable!("We made sure the shape was correct."))
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
                .iter()
                .map(|(_, models)| models.iter().map(|_| graph.clone()).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        };

        let labels = labels.iter().map(|&l| if l { 1.0 } else { 0.0 }).collect::<Vec<f32>>();
        let mut full_training_data = previous_data.unwrap_or_else(|| {
            self.algorithms
                .iter()
                .map(|(_, models)| models.iter().map(|_| (Vec::new(), Vec::new())).collect::<Vec<_>>())
                .collect()
        });

        for e in 0..num_epochs {
            let training_data_size = full_training_data
                .iter()
                .map(|m| m.iter().map(|m| m.0.len()).sum::<usize>())
                .sum::<usize>();
            println!(
                "Starting Inner Epoch {}/{num_epochs} with dataset size: {training_data_size}",
                e + 1
            );

            let roc_score = {
                let predictions = self.predict(data, root);
                let predictions = Self::aggregate_predictions(&predictions).to_vec();
                roc_auc_score(&labels.clone(), &predictions).as_f32()
            };
            println!("Inner current ROC Score: {roc_score:.6}");

            let new_training_data = self.generate_training_data(&mut graphs, &labels);

            full_training_data = full_training_data
                .into_iter()
                .zip(new_training_data)
                .map(|(m_old, m_new)| {
                    m_old
                        .into_iter()
                        .zip(m_new)
                        .map(|((mut x_old, mut y_old), (mut x_new, mut y_new))| {
                            x_old.append(&mut x_new);
                            y_old.append(&mut y_new);
                            (x_old, y_old)
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

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
                            let properties = clusters
                                .par_iter()
                                .map(|c| {
                                    let (p, p_) = c.ratios();
                                    let mut properties = p.to_vec();
                                    properties.extend_from_slice(p_.as_ref());
                                    properties
                                })
                                .collect::<Vec<_>>();
                            ml_model
                                .predict(&properties)
                                .unwrap_or_else(|_| unreachable!("We made sure the shape was correct."))
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
                                // The roc-score function needs both classes represented so we add a
                                // couple of dummy values to the end of the vectors.
                                let mut y_true = labels[start..(start + cardinality)].to_vec();
                                y_true.push(1.0);
                                y_true.push(0.0);
                                let mut y_pred = vec![rating; cardinality];
                                y_pred.push(1.0);
                                y_pred.push(0.0);
                                roc_auc_score(&y_true, &y_pred).as_f32()
                            })
                            .collect::<Vec<_>>();
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
                        model
                            .train(train_x, train_y)
                            .unwrap_or_else(|_| unreachable!("We made sure the shape was correct."));
                    });
            });
    }
}
