//! Utilities for selecting the fastest CAKES KNN search algorithm for a given dataset and metric.

use rand::prelude::*;
use rayon::prelude::*;

use crate::{cakes::BatchedSearch, utils::MaxItem, Cluster, DistanceValue};

/// Measures the throughput (Queries per Second) of a CAKES algorithm on the given root cluster with the given metric.
///
/// This function runs the algorithm on the provided queries and measures the time taken to complete them. It uses only
/// a single thread for the measurement.
#[allow(clippy::cast_precision_loss, clippy::while_float)]
pub fn measure_throughput<Id, I, T, A, M, Alg>(
    root: &Cluster<Id, I, T, A>,
    metric: &M,
    queries: &[&I],
    alg: &Alg,
    min_time_secs: f64,
) -> f64
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
    Alg: BatchedSearch<Id, I, T, A, M> + ?Sized,
{
    let start = std::time::Instant::now();
    let mut total_queries = 0;
    while start.elapsed().as_secs_f64() < min_time_secs {
        let _results = queries
            .iter()
            .map(|&query| alg.search(root, metric, query))
            .collect::<Vec<_>>();
        total_queries += queries.len();
    }
    total_queries as f64 / start.elapsed().as_secs_f64()
}

/// Parallel version of [`measure_throughput`](measure_throughput).
#[allow(clippy::cast_precision_loss, clippy::while_float)]
pub fn par_measure_throughput<Id, I, T, A, M, Alg>(
    root: &Cluster<Id, I, T, A>,
    metric: &M,
    queries: &[&I],
    alg: &Alg,
    min_time_secs: f64,
) -> f64
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
    Alg: BatchedSearch<Id, I, T, A, M> + ?Sized + Send + Sync,
{
    let start = std::time::Instant::now();
    let mut total_queries = 0;
    while start.elapsed().as_secs_f64() < min_time_secs {
        let _results = queries
            .par_iter()
            .map(|&query| alg.search(root, metric, query))
            .collect::<Vec<_>>();
        total_queries += queries.len();
    }
    total_queries as f64 / start.elapsed().as_secs_f64()
}

/// Selects the fastest CAKES algorithm for the given dataset and metric.
#[allow(clippy::type_complexity)]
pub fn select_fastest_algorithm<Id, I, T, A, M>(
    root: &Cluster<Id, I, T, A>,
    metric: &M,
    n_queries: usize,
    k: usize,
    min_time_secs: f64,
) -> (Box<dyn BatchedSearch<Id, I, T, A, M>>, f64)
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    let queries = {
        let mut all_items = root.all_items();
        all_items.shuffle(&mut rand::rng());
        all_items
            .into_iter()
            .take(n_queries)
            .map(|(_, item)| item)
            .collect::<Vec<_>>()
    };

    let algorithms: Vec<Box<dyn BatchedSearch<Id, I, T, A, M>>> = vec![
        Box::new(crate::cakes::KnnDfs(k)),
        Box::new(crate::cakes::KnnBfs(k)),
        Box::new(crate::cakes::KnnRrnn(k)),
        Box::new(crate::cakes::KnnBranch(k)),
    ];

    let algs_throughputs = algorithms
        .into_iter()
        .map(|alg| {
            let throughput = measure_throughput(root, metric, &queries, alg.as_ref(), min_time_secs);
            (alg, throughput)
        })
        .collect::<Vec<_>>();

    algs_throughputs
        .into_iter()
        .max_by_key(|(_, throughput)| MaxItem((), *throughput))
        .unwrap_or_else(|| unreachable!("We created more than zero algorithms"))
}

/// Parallel version of [`select_fastest_algorithm`](select_fastest_algorithm).
#[allow(clippy::type_complexity)]
pub fn par_select_fastest_algorithm<Id, I, T, A, M>(
    root: &Cluster<Id, I, T, A>,
    metric: &M,
    n_queries: usize,
    k: usize,
    min_time_secs: f64,
) -> (Box<dyn BatchedSearch<Id, I, T, A, M> + Send + Sync>, f64)
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
{
    let queries = {
        let mut all_items = root.all_items();
        all_items.shuffle(&mut rand::rng());
        all_items
            .into_iter()
            .take(n_queries)
            .map(|(_, item)| item)
            .collect::<Vec<_>>()
    };

    let algorithms: Vec<Box<dyn BatchedSearch<Id, I, T, A, M> + Send + Sync>> = vec![
        Box::new(crate::cakes::KnnDfs(k)),
        Box::new(crate::cakes::KnnBfs(k)),
        Box::new(crate::cakes::KnnRrnn(k)),
        Box::new(crate::cakes::KnnBranch(k)),
    ];

    let algs_throughputs = algorithms
        .into_iter()
        .map(|alg| {
            let throughput = par_measure_throughput(root, metric, &queries, alg.as_ref(), min_time_secs);
            (alg, throughput)
        })
        .collect::<Vec<_>>();

    algs_throughputs
        .into_iter()
        .max_by_key(|(_, throughput)| MaxItem((), *throughput))
        .unwrap_or_else(|| unreachable!("We created more than zero algorithms"))
}
