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

#[allow(clippy::unwrap_used)]
fn compute_compression_costs<Id, I, T: DistanceValue, M: Fn(&I, &I) -> T>(
    ball: &Ball<Id, I, T, CompressionCosts<T>>,
    metric: &M,
) -> Option<CompressionCosts<T>> {
    let costs = if let Some([left, right]) = ball.children() {
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
    };
    Some(costs)
}

fn run_group<P: AsRef<std::path::Path>>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    dataset: &AnnDataset,
    base: &P,
    max_queries: usize,
    shuffle: bool,
) {
    let queries = dataset.read_test(base, shuffle).unwrap();
    let queries = queries[..(max_queries.min(queries.len()))].to_vec();

    let mut items = dataset.read_train(base, shuffle).unwrap();
    let metric = dataset.metric();

    group
        .throughput(criterion::Throughput::Elements(queries.len() as u64))
        .sample_size(10);

    let error = {
        let n_items = items.len();
        let criteria = |b: &Ball<_, _, _, ()>| b.cardinality() > n_items;
        let mut root = Ball::par_new_tree_with_indices(items, &metric, &criteria).unwrap();

        let id = BenchmarkId::new("knn-linear-k10", 1_usize);
        group.bench_function(id, |b| {
            b.iter_with_large_drop(|| abd_clam::cakes::KnnLinear(10).par_batch_search(&root, &metric, &queries))
        });

        items = root.take_subtree_items().into_iter().map(|(_, v)| v).collect();
        items.push(root.center().clone());

        10.0 * root.radius() / n_items as f32
    };

    let multipliers = [1, 2, 4, 8, 16, 32, 64];
    for &multiplier in &multipliers[..1] {
        // For quicker benches, only use the first multiplier
        let augmented_items = if multiplier == 1 {
            items.clone()
        } else {
            symagen::augmentation::augment_data(&items, multiplier, error)
        };

        let criteria = |_: &_| true;
        let to_prune_or_not_to_prune = |b: &Ball<_, _, _, _>| {
            b.annotation()
                .map_or(false, |&CompressionCosts { unitary, recursive, .. }| {
                    unitary <= recursive
                })
        };

        let mut root = Ball::par_new_tree_with_indices(augmented_items, &metric, &criteria).unwrap();

        for prune in [false, true] {
            let pruned_str = if prune {
                // Compute compression costs
                root.par_annotate(&|_, _| None, &compute_compression_costs, &metric);
                // Prune based on compression costs
                root.prune(&to_prune_or_not_to_prune);
                // Remove annotations to maybe save some memory
                root.par_clear_annotations();

                "pruned"
            } else {
                "unpruned"
            };

            for k in [10] {
                let algs: &[(&'static str, Box<dyn ParSearch<_, _, _, _, _>>)] = &[
                    ("dfs", Box::new(cakes::KnnDfs(k))),
                    ("bfs", Box::new(cakes::KnnBfs(k))),
                    // ("rrnn", cakes::KnnRrnn(k)),
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

    let shuffle = true;
    let max_queries = 100;

    let base = base_dir().unwrap();
    for dataset in &datasets[..3] {
        // For quicker benches, only use the first 3 datasets
        let mut group = c.benchmark_group(dataset.name());
        run_group(&mut group, dataset, &base, max_queries, shuffle);
        group.finish();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
