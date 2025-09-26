//! Benchmarks for CAKES

#![expect(missing_docs)]

use abd_clam::{
    cakes::{self, ParSearch},
    Ball, DistanceValue,
};
use criterion::{criterion_group, criterion_main, Criterion};
use rayon::prelude::*;

mod utils;

use utils::ann_benchmarks::{base_dir, AnnDataset};

fn run_group<I, T, M>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    root: &Ball<I, T>,
    metric: &M,
    queries: &[I],
) where
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
{
    group
        .throughput(criterion::Throughput::Elements(queries.len() as u64))
        .sample_size(10);

    for k in [10] {
        let id = format!("knn-linear-{k}");
        let items = root.all_items();
        group.bench_function(&id, |b| {
            b.iter_with_large_drop(|| {
                queries
                    .into_par_iter()
                    .map(|q| items.par_iter().map(|&x| (x, metric(q, x))).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            })
        });

        let algs: &[(&'static str, Box<dyn ParSearch<I, T, M>>)] = &[
            ("knn-dfs", Box::new(cakes::KnnDfs(k))),
            ("knn-bfs", Box::new(cakes::KnnBfs(k))),
            // ("knn-rrnn", cakes::KnnRrnn(k)),
        ];

        let mut oracles = Vec::new();
        for (name, alg) in algs {
            oracles = alg
                .par_batch_search(root, metric, queries)
                .into_iter()
                .map(|res| {
                    res.into_iter()
                        .max_by_key(|&(i, d)| abd_clam::utils::MaxItem(i, d))
                        .map(|(_, radius)| cakes::RnnChess(radius))
                        .unwrap()
                })
                .collect();

            let id = format!("{name}-{k}");
            group.bench_function(&id, |b| {
                b.iter_with_large_drop(|| alg.par_batch_search(root, metric, queries))
            });
        }

        let id = format!("rnn-oracle-{k}");
        group.bench_function(&id, |b| {
            b.iter_with_large_drop(|| {
                oracles
                    .par_iter()
                    .zip(queries.par_iter())
                    .map(|(oracle, query)| oracle.par_search(root, metric, query))
                    .collect::<Vec<_>>()
            })
        });

        core::mem::drop(oracles);
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let datasets = [
        // For the paper
        AnnDataset::FashionMnist,
        AnnDataset::Sift,
        AnnDataset::Glove25,
        // The rest
        // AnnDataset::Mnist,
        // AnnDataset::Gist,
        // AnnDataset::Glove50,
        // AnnDataset::Glove100,
        // AnnDataset::Glove200,
        // AnnDataset::DeepImage,
    ];

    let shuffle = true;
    let max_queries = 1000;

    let base = base_dir().unwrap();
    for dataset in &datasets {
        let items = dataset.read_train(&base, shuffle).unwrap();
        let queries = dataset.read_test(&base, shuffle).unwrap();
        let queries = queries[..(max_queries.min(queries.len()))].to_vec();
        let metric = dataset.metric();

        let criteria = |_: &Ball<_, _>| true;
        // let criteria = |b: &Ball<_, _>| b.cardinality() > 10;
        let root = Ball::par_new_tree(items, &metric, &criteria).unwrap();

        let mut group = c.benchmark_group(dataset.name());
        run_group(&mut group, &root, &metric, &queries);
        group.finish();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
