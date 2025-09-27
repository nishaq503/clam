//! Entropy Scaling Search

use rayon::prelude::*;

use crate::{Ball, DistanceValue};

mod knn_bfs;
mod knn_dfs;
mod knn_linear;
mod knn_rrnn;
mod rnn_chess;
mod rnn_linear;

pub use knn_bfs::KnnBfs;
pub use knn_dfs::KnnDfs;
pub use knn_linear::KnnLinear;
pub use knn_rrnn::KnnRrnn;
pub use rnn_chess::RnnChess;
pub use rnn_linear::RnnLinear;

/// A `Search` trait for defining how to search for nearest neighbors.
pub trait Search<Id, I, T: DistanceValue, M: Fn(&I, &I) -> T, A> {
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
    fn search<'a>(&self, root: &'a Ball<Id, I, T, A>, metric: &M, query: &I) -> Vec<(&'a Id, &'a I, T)>;

    /// Batched version of [`Search::search`](Search::search).
    fn batch_search<'a>(&self, root: &'a Ball<Id, I, T, A>, metric: &M, queries: &[I]) -> Vec<Vec<(&'a Id, &'a I, T)>> {
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
    fn par_search<'a>(&self, root: &'a Ball<Id, I, T, A>, metric: &M, query: &I) -> Vec<(&'a Id, &'a I, T)> {
        self.search(root, metric, query)
    }

    /// Parallel batched version of [`ParSearch::par_search`](ParSearch::par_search).
    fn par_batch_search<'a>(
        &self,
        root: &'a Ball<Id, I, T, A>,
        metric: &M,
        queries: &[I],
    ) -> Vec<Vec<(&'a Id, &'a I, T)>> {
        queries
            .par_iter()
            .map(|query| self.par_search(root, metric, query))
            .collect()
    }
}
