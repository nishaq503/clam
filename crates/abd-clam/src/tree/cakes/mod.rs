//! Entropy Scaling Search

use rayon::prelude::*;

use crate::DistanceValue;

use super::Tree;

mod exact;

// pub(crate) use exact::{leaf_into_hits, pop_till_leaf};
pub use exact::{KnnBfs, KnnBranch, KnnDfs, KnnLinear, KnnRrnn, RnnChess, RnnLinear};

/// A Nearest Neighbor Search algorithm.
pub trait Search<Id, I, T, A, M>: std::fmt::Display
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    /// Searches for nearest neighbors of `query` in the given `tree` and returns a vector of `(index, distance)` pairs into the `items` of the `tree`.
    fn search(&self, tree: &Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)>;

    /// Parallel version of [`Search::search`].
    ///
    /// The default implementation offers no parallelism. This method should be overridden for algorithms that will actually benefit from parallelism when
    /// searching for a single query.
    fn par_search(&self, tree: &Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)>
    where
        Self: Send + Sync,
        Id: Send + Sync,
        I: Send + Sync,
        T: Send + Sync,
        M: Send + Sync,
        A: Send + Sync,
    {
        self.search(tree, query)
    }

    /// Batched version of [`Search::search`].
    fn batch_search(&self, tree: &Tree<Id, I, T, A, M>, queries: &[I]) -> Vec<Vec<(usize, T)>> {
        queries.iter().map(|query| self.search(tree, query)).collect()
    }

    /// Parallel version of [`Search::batch_search`].
    fn par_batch_search(&self, tree: &Tree<Id, I, T, A, M>, queries: &[I]) -> Vec<Vec<(usize, T)>>
    where
        Self: Send + Sync,
        Id: Send + Sync,
        I: Send + Sync,
        T: Send + Sync,
        M: Send + Sync,
        A: Send + Sync,
    {
        queries.par_iter().map(|query| self.search(tree, query)).collect()
    }

    /// Parallel batched version of [`Search::par_search`].
    fn par_batch_par_search(&self, tree: &Tree<Id, I, T, A, M>, queries: &[I]) -> Vec<Vec<(usize, T)>>
    where
        Self: Send + Sync,
        Id: Send + Sync,
        I: Send + Sync,
        T: Send + Sync,
        M: Send + Sync,
        A: Send + Sync,
    {
        queries.par_iter().map(|query| self.par_search(tree, query)).collect()
    }
}

/// The minimum possible distance from the query to any item in the cluster.
pub(crate) fn d_min<T: DistanceValue, A>(node: &super::Node<T, A>, d: T) -> T {
    if d < node.radius() {
        T::zero()
    } else {
        d - node.radius()
    }
}

/// Returns the theoretical maximum distance from the query to a point in the cluster.
pub(crate) fn d_max<T: DistanceValue, A>(node: &super::Node<T, A>, d: T) -> T {
    node.radius() + d
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
pub fn search_quality_stats<T: DistanceValue>(
    true_hits: &[Vec<(usize, T)>],
    pred_hits: &[Vec<(usize, T)>],
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
fn compute_recall_single<T: DistanceValue>(true_hits: &[(usize, T)], pred_hits: &[(usize, T)]) -> f64 {
    let max_distance = true_hits.last().unwrap().1;
    let n_valid_hits = pred_hits.iter().filter(|&&(_, d)| d <= max_distance).count();
    n_valid_hits as f64 / true_hits.len() as f64
}

/// Sorts the search results by distance in non-decreasing order.
fn sorted_by_distance<T: DistanceValue>(hits: &[(usize, T)]) -> Vec<(usize, T)> {
    let mut hits = hits.to_vec();
    hits.sort_by_key(|&(_, dist)| crate::utils::MinItem((), dist));
    hits
}

/// Computes the distance-error of pairs of true and predicted nearest neighbor search results.
#[expect(clippy::cast_precision_loss, clippy::unwrap_used)]
fn compute_distance_error<T: DistanceValue>(true_hits: &[(usize, T)], pred_hits: &[(usize, T)]) -> f64 {
    let err_sum = true_hits
        .iter()
        .zip(pred_hits.iter())
        .map(|((_, d_true), (_, d_pred))| {
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
