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
                .inspect(|member| println!("Member: {}", member.name()))
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

        let num_predictions = predictions.len();
        let predictions = predictions.into_iter().flatten().collect::<Vec<_>>();
        Array2::from_shape_vec((num_predictions, data.cardinality()), predictions)
            .unwrap_or_else(|_| unreachable!("We made sure the shape was correct."))
    }

    /// Aggregate the predictions of the ensemble.
    ///
    /// For now, we take the mean of the anomaly scores for each point. Later,
    /// we may want to consider other aggregation methods.
    #[must_use]
    pub fn aggregate_predictions(scores: &Array2<f32>) -> Vec<f32> {
        // Take the mean of the anomaly scores for each point
        scores
            .mean_axis(Axis(0))
            .unwrap_or_else(|| unreachable!("We made sure the shape was correct."))
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
            println!(
                "Using previous data for training on {} over {num_epochs} epochs.",
                data.name()
            );
            self.create_graphs(data, root)
        } else {
            println!("Training from scratch on {} over {num_epochs} epochs.", data.name());
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
            println!("Initial graph created.");
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
                            let shape = (anomaly_properties.len(), anomaly_properties[0].len());
                            let properties = anomaly_properties.into_iter().flatten().collect::<Vec<_>>();
                            let properties = Array2::from_shape_vec(shape, properties)
                                .unwrap_or_else(|_| unreachable!("We made sure the shape was correct."));
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
                        let shape = (train_x.len(), train_x[0].len());
                        let train_x = train_x.iter().flatten().copied().collect::<Vec<_>>();
                        let train_x = Array2::from_shape_vec(shape, train_x)
                            .map_err(|e| e.to_string())
                            .unwrap_or_else(|e| unreachable!("{e}"));
                        model
                            .train(&train_x, train_y)
                            .unwrap_or_else(|_| unreachable!("We made sure the shape was correct."));
                    });
            });
    }
}
