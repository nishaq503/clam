//! Benchmarks for CAKES

#![expect(missing_docs)]

use abd_clam::{
    cakes::{self, ParSearch},
    Ball, DistanceValue,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rayon::prelude::*;

mod utils;

use utils::ann_benchmarks::{base_dir, AnnDataset};

struct CompressionCosts<T: DistanceValue> {
    recursive: T,
    unitary: T,
    minimum: T,
}

fn bench_for_ks<Id, I, T, A, M>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    root: &Ball<Id, I, T, A>,
    metric: &M,
    queries: &[I],
    pruned: bool,
    multiplier: usize,
    ks: &[usize],
) where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
{
    let pruned_str = if pruned { "pruned" } else { "unpruned" };

    for &k in ks {
        let algs: &[(&'static str, Box<dyn ParSearch<_, _, _, _, _>>)] = &[
            ("dfs", Box::new(cakes::KnnDfs(k))),
            ("bfs", Box::new(cakes::KnnBfs(k))),
            // ("rrnn", Box::new(cakes::KnnRrnn(k))),
        ];

        let mut oracles = Vec::new();
        for (name, alg) in algs {
            oracles = alg
                .par_batch_search(&root, &metric, &queries)
                .into_iter()
                .map(|res| {
                    res.into_iter()
                        .max_by_key(|&(_, _, d)| abd_clam::utils::MaxItem((), d))
                        .map(|(_, _, radius)| cakes::RnnChess(radius))
                        .unwrap()
                })
                .collect();

            let id = BenchmarkId::new(format!("{name}-k{k}-{pruned_str}"), multiplier);
            group.bench_function(id, |b| {
                b.iter_with_large_drop(|| alg.par_batch_search(&root, &metric, &queries))
            });
        }

        let id = BenchmarkId::new(format!("oracle-k{k}-{pruned_str}"), multiplier);
        group.bench_function(id, |b| {
            b.iter_with_large_drop(|| {
                oracles
                    .par_iter()
                    .zip(queries.par_iter())
                    .map(|(oracle, query)| oracle.par_search(&root, &metric, query))
                    .collect::<Vec<_>>()
            })
        });

        core::mem::drop(oracles);
    }
}

fn run_group<P: AsRef<std::path::Path>>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    dataset: &AnnDataset,
    base: &P,
    max_items: usize,
    max_queries: usize,
    shuffle: bool,
    ks: &[usize],
) {
    let items = dataset.read_train(base, shuffle).unwrap();
    let metric = dataset.metric();
    let queries = dataset.read_test(base, shuffle).unwrap();
    let queries = queries[..(max_queries.min(queries.len()))].to_vec();

    // Function to compute compression costs for a ball in a post-order traversal
    let compute_compression_costs = |ball: &Ball<_, _, _, CompressionCosts<_>>| {
        if let Some([left, right]) = ball.children() {
            // INVARIANT: This function is called in post-order, so the children's costs are already computed.
            let recursive = left.annotation().unwrap().minimum
                + right.annotation().unwrap().minimum
                + metric(left.center(), ball.center())
                + metric(right.center(), ball.center());
            let unitary = ball.radial_sum();
            let minimum = if recursive < unitary { recursive } else { unitary };

            CompressionCosts {
                recursive,
                unitary,
                minimum,
            }
        } else {
            CompressionCosts {
                recursive: ball.radial_sum(),
                unitary: ball.radial_sum(),
                minimum: ball.radial_sum(),
            }
        }
    };
    // Predicate to decide whether to prune a ball based on its compression costs
    let to_prune_or_not_to_prune = |b: &Ball<_, _, _, _>| {
        b.annotation()
            .map_or(false, |&CompressionCosts { unitary, recursive, .. }| {
                unitary <= recursive
            })
    };

    group
        .throughput(criterion::Throughput::Elements(queries.len() as u64))
        .sample_size(10);

    let augmentation_error = {
        let n_items = items.len();
        let root = Ball::par_new_tree_minimal(items.clone(), &metric, &|b: _| b.cardinality() > n_items).unwrap();

        // Baseline: linear scan with k=10 on the original data only
        let id = BenchmarkId::new("knn-linear-k10", 1_usize);
        group.bench_function(id, |b| {
            b.iter_with_large_drop(|| abd_clam::cakes::KnnLinear(10).par_batch_search(&root, &metric, &queries))
        });

        // augmentation error: 0.1% of the radius of the root ball
        root.radius() / 1000.0
    };

    for multiplier in (0..).map(|p| 2_usize.pow(p)) {
        if multiplier > 1 && items.len() * multiplier > max_items {
            // Stop if augmentation would exceed max_items
            break;
        }

        let augmented_items = if multiplier == 1 {
            items.clone()
        } else {
            symagen::augmentation::augment_data(&items, multiplier, augmentation_error)
        };

        // Build a tree with no annotations and benchmark the search algorithms
        let root = Ball::par_new_tree_minimal(augmented_items, &metric, &|_| true).unwrap();
        bench_for_ks(group, &root, &metric, &queries, false, multiplier, ks);

        // Annotate the tree with compression costs, prune it to the necessarily unitary leaves, and benchmark the search algorithms again
        let root = {
            let mut root = root.par_reset_annotations(); // We first have to change the annotation type
            root.par_annotate_post_order(&compute_compression_costs); // Compute compression costs
            root.prune(&to_prune_or_not_to_prune); // Prune based on compression costs
            root.par_reset_annotations::<()>() // We don't need the annotations anymore, so we clear them to save memory
        };
        bench_for_ks(group, &root, &metric, &queries, true, multiplier, ks);
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let datasets = [
        // For the paper
        AnnDataset::FashionMnist,
        AnnDataset::Sift,
        AnnDataset::Glove25,
        // The rest
        AnnDataset::Mnist,
        AnnDataset::Gist,
        AnnDataset::Glove50,
        AnnDataset::Glove100,
        AnnDataset::Glove200,
        AnnDataset::DeepImage,
    ];

    // Whether to shuffle the dataset before building the tree
    let shuffle = true;

    // Set these parameters to control the runtime of the benchmarks
    let max_items = 32_000_000;
    let max_queries = 100;
    let ks = [10];

    let base = base_dir().unwrap();
    for dataset in &datasets[..3] {
        // For the paper, only use the first 3 datasets
        let mut group = c.benchmark_group(dataset.name());
        run_group(&mut group, dataset, &base, max_items, max_queries, shuffle, &ks);
        group.finish();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
