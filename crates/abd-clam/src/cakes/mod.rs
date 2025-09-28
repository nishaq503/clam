//! Entropy Scaling Search

use rayon::prelude::*;

use crate::{Cluster, DistanceValue};

pub mod approximate;
mod exact;

pub use exact::{KnnBfs, KnnDfs, KnnLinear, KnnRrnn, RnnChess, RnnLinear};

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

/// Computes statistics about the quality of approximate nearest neighbor search results.
///
/// These include:
///  - Minimum recall
///  - Maximum recall
///  - Mean recall
///  - Standard deviation of recall
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
/// - If any of the distance values cannot be converted to `f64`.
#[expect(clippy::cast_precision_loss)]
#[must_use]
pub fn search_quality_stats<Id, I, T: DistanceValue>(
    true_hits: &[Vec<(&Id, &I, T)>],
    pred_hits: &[Vec<(&Id, &I, T)>],
) -> Vec<(&'static str, f64)> {
    assert_eq!(true_hits.len(), pred_hits.len());
    assert!(!true_hits.is_empty());

    let recalls = true_hits
        .iter()
        .zip(pred_hits.iter())
        .map(|(true_hit, approx_hit)| compute_recall_single(true_hit, approx_hit))
        .collect::<Vec<_>>();

    let (min_recall, max_recall, sum_recall) = recalls
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY, 0.0), |(min, max, sum), &r| {
            (min.min(r), max.max(r), sum + r)
        });

    let mean_recall = sum_recall / recalls.len() as f64;
    let std_dev = (recalls.iter().map(|&r| (r - mean_recall).powi(2)).sum::<f64>() / recalls.len() as f64).sqrt();
    vec![
        ("recall_min", min_recall),
        ("recall_max", max_recall),
        ("recall_mean", mean_recall),
        ("recall_std_dev", std_dev),
    ]
}

/// Computes the recall of approximate nearest neighbor search results for a single query.
#[expect(clippy::unwrap_used)]
fn compute_recall_single<Id, I, T: DistanceValue>(true_hits: &[(&Id, &I, T)], pred_hits: &[(&Id, &I, T)]) -> f64 {
    if true_hits.is_empty() {
        if pred_hits.is_empty() {
            1.0
        } else {
            0.0
        }
    } else if pred_hits.is_empty() {
        0.0
    } else {
        let true_hits = sorted_by_distance(true_hits);
        let pred_hits = sorted_by_distance(pred_hits);

        let mut true_hits = true_hits.into_iter().map(|(_, _, d)| d);
        let mut pred_hits = pred_hits.into_iter().map(|(_, _, d)| d);

        let mut recall = 0.0;
        let mut count = 0.0;
        loop {
            match (true_hits.next(), pred_hits.next()) {
                (Some(t), Some(p)) if ((t - p).to_f64().unwrap().abs() < f64::EPSILON.sqrt().sqrt()) => {
                    recall += 1.0;
                    count += 1.0;
                }
                (Some(_), Some(_) | None) => {
                    count += 1.0;
                }
                (None, _) => break,
            }
        }

        recall / count
    }
}

/// Sorts the search results by distance in non-decreasing order.
fn sorted_by_distance<'a, Id, I, T: DistanceValue>(hits: &[(&'a Id, &'a I, T)]) -> Vec<(&'a Id, &'a I, T)> {
    let mut hits = hits.to_vec();
    hits.sort_by_key(|&(_, _, dist)| crate::utils::MinItem((), dist));
    hits
}
