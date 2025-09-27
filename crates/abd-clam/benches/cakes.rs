//! Benchmarks for CAKES

#![expect(missing_docs)]

use abd_clam::{
    cakes::{self, ParSearch},
    Ball,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rayon::prelude::*;

mod utils;

use utils::ann_benchmarks::{base_dir, AnnDataset};

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
    // dataset.normalize_if_cosine(&mut items);
    let metric = dataset.metric();

    group
        .throughput(criterion::Throughput::Elements(queries.len() as u64))
        .sample_size(10);

    let error = {
        let n_items = items.len();
        let criteria = |b: &Ball<_, _, _>| b.cardinality() > n_items;
        let mut root = Ball::par_new_tree_with_indices(items, &metric, &criteria).unwrap();

        let id = BenchmarkId::new("knn-linear-k10", 1_usize);
        group.bench_function(id, |b| {
            b.iter_with_large_drop(|| abd_clam::cakes::KnnLinear(10).par_batch_search(&root, &metric, &queries))
        });

        items = root.take_subtree_items().into_iter().map(|(_, v)| v).collect();
        items.push(root.center().1.clone());

        10.0 * root.radius() / n_items as f32
    };

    let multipliers = [1, 2, 4, 8, 16, 32, 64];
    for &multiplier in &multipliers[..1] {
        // For quicker benches, only use the first multiplier
        let augmented_items = if multiplier == 1 {
            items.clone()
        } else {
            #[allow(unused_mut)]
            let mut augmented_items = symagen::augmentation::augment_data(&items, multiplier, error);
            // dataset.normalize_if_cosine(&mut augmented_items);
            augmented_items
        };

        let criteria = |_: &Ball<_, _, _>| true;
        let root = Ball::par_new_tree_with_indices(augmented_items, &metric, &criteria).unwrap();

        for k in [10] {
            let algs: &[(&'static str, Box<dyn ParSearch<_, _, _, _>>)] = &[
                ("knn-dfs", Box::new(cakes::KnnDfs(k))),
                ("knn-bfs", Box::new(cakes::KnnBfs(k))),
                // ("knn-rrnn", cakes::KnnRrnn(k)),
            ];

            let mut oracles = Vec::new();
            for (name, alg) in algs {
                oracles = alg
                    .par_batch_search(&root, &metric, &queries)
                    .into_iter()
                    .map(|res| {
                        res.into_iter()
                            .max_by_key(|&(_, d)| abd_clam::utils::MaxItem((), d))
                            .map(|(_, radius)| cakes::RnnChess(radius))
                            .unwrap()
                    })
                    .collect();

                let id = BenchmarkId::new(format!("{name}-k{k}"), multiplier);
                group.bench_function(id, |b| {
                    b.iter_with_large_drop(|| alg.par_batch_search(&root, &metric, &queries))
                });
            }

            let id = BenchmarkId::new(format!("rnn-oracle-k{k}"), multiplier);
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
