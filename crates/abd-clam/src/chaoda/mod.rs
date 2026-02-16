//! Anomaly detection algorithms using CLAM.
//!
//! This module contains CLAM-CHAODA (Clustered Hierarchical Anomaly and Outlier Detection Algorithms). This is a family of algorithms that use CLAM trees to
//! impute graphs that enable unsupervised anomaly detection algorithms.
//!
//! For trees, this enables the [`Tree::annotate_anomaly_features`](crate::Tree::annotate_anomaly_features) method (along with its parallel version). These
//! features can then be used for creating CHAODA graphs. The graphs can, in turn, be used for anomaly detection using the algorithms we provide...

mod algorithms;
mod graph;
mod learning;
mod tree;

pub use algorithms::ChaodaAlgorithm;
pub use graph::{Component, Graph, Node};
pub use learning::{AnomalyFeatures, MetaMlModel, metrics};
