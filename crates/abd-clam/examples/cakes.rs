//! Example for running CAKES search.

use std::path::Path;

use abd_clam::{cakes::ParSearch, Ball};
use rayon::prelude::*;

fn build_metric(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    distances::simd::euclidean_f32(a, b)
}

fn knn_dfs_metric(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    profi::prof!();
    distances::simd::euclidean_f32(a, b)
}

fn knn_bfs_metric(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    profi::prof!();
    distances::simd::euclidean_f32(a, b)
}

fn knn_rrnn_metric(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    profi::prof!();
    distances::simd::euclidean_f32(a, b)
}

fn oracle_metric(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    profi::prof!();
    distances::simd::euclidean_f32(a, b)
}

fn read_data(name: &str) -> Result<[Vec<Vec<f32>>; 2], String> {
    let base = Path::new(".")
        .canonicalize()
        .map_err(|e| e.to_string())?
        .parent()
        .ok_or_else(|| "Failed to get parent directory.".to_string())?
        .join("data/ann_data");

    let data = read_array(base.join(format!("{}-train.npy", name)))?;
    let queries = read_array(base.join(format!("{}-test.npy", name)))?;

    Ok([data, queries])
}

fn read_array<P: AsRef<Path>>(path: P) -> Result<Vec<Vec<f32>>, String> {
    let array = ndarray_npy::read_npy::<_, ndarray::Array2<f32>>(path).map_err(|e| e.to_string())?;
    Ok(array.outer_iter().map(|row| row.to_vec()).collect())
}

fn build_ball(items: Vec<Vec<f32>>) -> Result<Ball<usize, Vec<f32>, f32, ()>, String> {
    Ball::par_new_tree_with_indices(items, &build_metric, &|_| true)
}

fn search_ball<'a>(root: &'a Ball<usize, Vec<f32>, f32, ()>, queries: &[Vec<f32>], k: usize) {
    profi::prof!();

    let dfs_results = abd_clam::cakes::KnnDfs(k).par_batch_search(root, &knn_dfs_metric, queries);
    let bfs_results = abd_clam::cakes::KnnBfs(k).par_batch_search(root, &knn_bfs_metric, queries);
    let rrnn_results = abd_clam::cakes::KnnRrnn(k).par_batch_search(root, &knn_rrnn_metric, queries);

    let oracles = dfs_results
        .iter()
        .map(|res| {
            res.iter()
                .max_by_key(|&&(_, d)| abd_clam::utils::MaxItem((), d))
                .map(|&(_, radius)| abd_clam::cakes::RnnChess(radius))
                .unwrap()
        })
        .collect::<Vec<_>>();

    let oracle_results = oracles
        .into_par_iter()
        .zip(queries)
        .map(|(oracle, query)| oracle.par_search(root, &oracle_metric, query))
        .collect::<Vec<_>>();

    for (i, (res, oracle)) in dfs_results.into_iter().zip(oracle_results).enumerate() {
        assert_eq!(res.len(), k, "Query {i} expected {k} results, got {}", res.len());
        assert!(
            oracle.len() >= k,
            "Query {i} expected at least {k} oracle results, got {}",
            oracle.len()
        );
    }

    core::mem::drop(bfs_results);
    core::mem::drop(rrnn_results);
}

fn main() -> Result<(), String> {
    let mut file = std::fs::File::create("./logs/profi-cakes.txt").map_err(|e| e.to_string())?;
    profi::print_on_exit!(to = &mut file);

    let k = 10;

    let [items, queries] = read_data("fashion-mnist")?;
    let root = build_ball(items)?;

    search_ball(&root, &queries[..1000], k);

    Ok(())
}
