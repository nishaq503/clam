//! Example for running CAKES search.

use std::path::Path;

use abd_clam::{cakes::ParSearch, Ball};

fn build_metric(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    distances::simd::euclidean_f32(a, b)
}

fn search_metric(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    profi::prof!();
    distances::simd::euclidean_f32(a, b)
}

fn read_sift() -> Result<[Vec<Vec<f32>>; 2], String> {
    let base = Path::new(".")
        .canonicalize()
        .map_err(|e| e.to_string())?
        .parent()
        .ok_or_else(|| "Failed to get parent directory.".to_string())?
        .join("data/ann_data");

    let data = read_array(base.join("sift-train.npy"))?;
    let queries = read_array(base.join("sift-test.npy"))?;

    Ok([data, queries])
}

fn read_array<P: AsRef<Path>>(path: P) -> Result<Vec<Vec<f32>>, String> {
    let array = ndarray_npy::read_npy::<_, ndarray::Array2<f32>>(path).map_err(|e| e.to_string())?;
    Ok(array.outer_iter().map(|row| row.to_vec()).collect())
}

fn build_ball(items: Vec<Vec<f32>>) -> Result<Ball<Vec<f32>, f32>, String> {
    Ball::par_new_tree(items, &build_metric, &|_| true)
}

fn search_ball<'a>(
    root: &'a Ball<Vec<f32>, f32>,
    queries: &[Vec<f32>],
    k: usize,
    trimmed: bool,
) -> Vec<Vec<(&'a Vec<f32>, f32)>> {
    profi::prof!(if trimmed { "trimmed-search" } else { "full-search" });

    let alg = abd_clam::cakes::KnnDfs(k);
    alg.par_batch_search(root, &search_metric, queries)
}

fn main() -> Result<(), String> {
    let mut file = std::fs::File::create("./logs/profi-cakes.txt").map_err(|e| e.to_string())?;
    profi::print_on_exit!(to = &mut file);

    let k = 10;

    let [items, queries] = read_sift()?;
    let mut root = build_ball(items)?;

    let results = search_ball(&root, &queries[..100], k, false); // Full search.
    for res in results {
        assert_eq!(res.len(), k);
    }

    root.trim(&|b: &Ball<_, _>| b.cardinality() < 10); // Trim the tree.
    let results = search_ball(&root, &queries[..100], k, true); // Trimmed search.
    for res in results {
        assert_eq!(res.len(), k);
    }

    Ok(())
}
