//! Entropy Scaling Search

use rayon::prelude::*;

use crate::{Cluster, DistanceValue};

pub mod approximate;
mod exact;

pub(crate) use exact::{leaf_into_hits, pop_till_leaf};
pub use exact::{KnnBfs, KnnBranch, KnnDfs, KnnLinear, KnnRrnn, RnnChess, RnnLinear};

/// A `Search` trait for defining how to search for nearest neighbors.
pub trait Search<Id, I, T: DistanceValue, M: Fn(&I, &I) -> T, A>: std::fmt::Display {
    /// Search for the nearest neighbors of a given query item.
    ///
    /// # Arguments
    ///
    /// * `query` - The item to search for.
    /// * `k` - The number of nearest neighbors to find.
    ///
    /// # Returns
    ///
    /// A vector of tuples containing the index and distance of the nearest neighbors.
    fn search<'a>(&self, root: &'a Cluster<Id, I, T, A>, metric: &M, query: &I) -> Vec<(&'a Id, &'a I, T)>;

    /// Batched version of [`Search::search`](Search::search).
    fn batch_search<'a>(
        &self,
        root: &'a Cluster<Id, I, T, A>,
        metric: &M,
        queries: &[I],
    ) -> Vec<Vec<(&'a Id, &'a I, T)>> {
        queries.iter().map(|query| self.search(root, metric, query)).collect()
    }
}

/// A parallel extension of the [`Search`](Search) trait.
pub trait ParSearch<
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
    A: Send + Sync,
>: Search<Id, I, T, M, A> + Send + Sync
{
    /// Parallel version of [`Search::search`](Search::search).
    fn par_search<'a>(&self, root: &'a Cluster<Id, I, T, A>, metric: &M, query: &I) -> Vec<(&'a Id, &'a I, T)> {
        self.search(root, metric, query)
    }

    /// Parallel batched version of [`ParSearch::par_search`](ParSearch::par_search).
    fn par_batch_search<'a>(
        &self,
        root: &'a Cluster<Id, I, T, A>,
        metric: &M,
        queries: &[I],
    ) -> Vec<Vec<(&'a Id, &'a I, T)>> {
        queries
            .par_iter()
            .map(|query| self.par_search(root, metric, query))
            .collect()
    }
}

/// The minimum possible distance from the query to any item in the cluster.
pub(crate) fn d_min<Id, I, T: DistanceValue, A>(cluster: &Cluster<Id, I, T, A>, d: T) -> T {
    if d < cluster.radius() {
        T::zero()
    } else {
        d - cluster.radius()
    }
}

/// Returns the theoretical maximum distance from the query to a point in the cluster.
pub(crate) fn d_max<Id, I, T: DistanceValue, A>(cluster: &Cluster<Id, I, T, A>, d: T) -> T {
    cluster.radius() + d
}

/// Computes summary statistics about the quality of approximate nearest neighbor search results.
///
/// # Arguments
///
/// * `true_hits` - A slice of vectors containing the true nearest neighbors for each query, usually obtained from linear search or an exact search method.
/// * `pred_hits` - A slice of vectors containing the predicted nearest neighbors for each query.
///
/// # Returns
///
/// A vector of tuples containing the name and value of each statistic.
///
/// # Panics
///
/// - If `true_hits` is empty.
/// - If the lengths of `true_hits` and `pred_hits` do not match.
/// - If any of the inner vectors in `true_hits` is empty.
/// - If any pair of inner vectors in `true_hits` and `pred_hits` do not have the same length.
/// - If any of the distance values cannot be converted to `f64`.
#[must_use]
pub fn search_quality_stats<Id, I, T: DistanceValue>(
    true_hits: &[Vec<(&Id, &I, T)>],
    pred_hits: &[Vec<(&Id, &I, T)>],
) -> Vec<(String, f64)> {
    assert_eq!(true_hits.len(), pred_hits.len());
    assert!(!true_hits.is_empty());
    assert!(true_hits.iter().all(|v| !v.is_empty()));
    assert!(true_hits.iter().zip(pred_hits.iter()).all(|(a, b)| a.len() == b.len()));

    let true_hits = true_hits.iter().map(|v| sorted_by_distance(v)).collect::<Vec<_>>();
    let pred_hits = pred_hits.iter().map(|v| sorted_by_distance(v)).collect::<Vec<_>>();

    let recalls = true_hits
        .iter()
        .zip(pred_hits.iter())
        .map(|(true_hit, approx_hit)| compute_recall_single(true_hit, approx_hit))
        .collect::<Vec<_>>();
    let recall_stats = compute_summary_stats(&recalls);

    let d_errs = true_hits
        .iter()
        .zip(pred_hits.iter())
        .map(|(true_hit, approx_hit)| compute_distance_error(true_hit, approx_hit))
        .collect::<Vec<_>>();
    let d_err_stats = compute_summary_stats(&d_errs);

    recall_stats
        .into_iter()
        .map(|(name, value)| (format!("{name} recall"), value))
        .chain(
            d_err_stats
                .into_iter()
                .map(|(name, value)| (format!("{name} d_err "), value)),
        )
        .collect()
}

/// Computes basic summary statistics for a slice of f64 values.
///
/// These include:
///
/// - Minimum
/// - Maximum
/// - Mean
/// - Standard Deviation
#[expect(clippy::cast_precision_loss)]
fn compute_summary_stats(values: &[f64]) -> Vec<(&'static str, f64)> {
    let (min, max, sum) = values
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY, 0.0), |(min, max, sum), &v| {
            (min.min(v), max.max(v), sum + v)
        });
    let mean = sum / values.len() as f64;
    let std_dev = (values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt();
    vec![
        ("min    ", min),
        ("max    ", max),
        ("mean   ", mean),
        ("std_dev", std_dev),
    ]
}

/// Computes the recall of approximate nearest neighbor search results for a single query.
///
/// The `true_hits` and `pred_hits` slices should be sorted by distance in non-decreasing order.
#[expect(clippy::cast_precision_loss, clippy::unwrap_used)]
fn compute_recall_single<Id, I, T: DistanceValue>(true_hits: &[(&Id, &I, T)], pred_hits: &[(&Id, &I, T)]) -> f64 {
    let max_distance = true_hits.last().unwrap().2;
    let n_valid_hits = pred_hits.iter().filter(|&&(_, _, d)| d <= max_distance).count();
    n_valid_hits as f64 / true_hits.len() as f64
}

/// Sorts the search results by distance in non-decreasing order.
fn sorted_by_distance<'a, Id, I, T: DistanceValue>(hits: &[(&'a Id, &'a I, T)]) -> Vec<(&'a Id, &'a I, T)> {
    let mut hits = hits.to_vec();
    hits.sort_by_key(|&(_, _, dist)| crate::utils::MinItem((), dist));
    hits
}

/// Computes the distance-error of pairs of true and predicted nearest neighbor search results.
#[expect(clippy::cast_precision_loss, clippy::unwrap_used)]
fn compute_distance_error<Id, I, T: DistanceValue>(true_hits: &[(&Id, &I, T)], pred_hits: &[(&Id, &I, T)]) -> f64 {
    let err_sum = true_hits
        .iter()
        .zip(pred_hits.iter())
        .map(|((_, _, d_true), (_, _, d_pred))| {
            let d_true = d_true.to_f64().unwrap();
            let d_pred = d_pred.to_f64().unwrap();
            if d_true == 0.0 || d_pred == 0.0 {
                0.0
            } else {
                d_pred / d_true - 1.0
            }
        })
        .sum::<f64>();
    err_sum / true_hits.len() as f64
}
