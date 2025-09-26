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

fn run_group<M>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    mut items: Vec<Vec<f32>>,
    metric: &M,
    queries: &[Vec<f32>],
) where
    M: Fn(&Vec<f32>, &Vec<f32>) -> f32 + Send + Sync,
{
    group
        .throughput(criterion::Throughput::Elements(queries.len() as u64))
        .sample_size(10);

    let error = {
        let n_items = items.len();
        let criteria = |b: &Ball<_, _>| b.cardinality() > n_items;
        let mut root = Ball::par_new_tree(items, &metric, &criteria).unwrap();

        let id = BenchmarkId::new("knn-linear-k10", 1_usize);
        group.bench_function(id, |b| {
            b.iter_with_large_drop(|| abd_clam::cakes::KnnLinear(10).par_batch_search(&root, metric, queries))
        });

        items = root.take_subtree_items();
        items.push(root.center().clone());

        10.0 * root.radius() / n_items as f32
    };

    for multiplier in [1, 2, 4, 8, 16, 32, 64] {
        let m_items = if multiplier == 1 {
            items.clone()
        } else {
            symagen::augmentation::augment_data(&items, multiplier, error)
        };

        let criteria = |_: &Ball<_, _>| true;
        let root = Ball::par_new_tree(m_items, &metric, &criteria).unwrap();

        for k in [10] {
            let algs: &[(&'static str, Box<dyn ParSearch<_, _, M>>)] = &[
                ("knn-dfs", Box::new(cakes::KnnDfs(k))),
                ("knn-bfs", Box::new(cakes::KnnBfs(k))),
                // ("knn-rrnn", cakes::KnnRrnn(k)),
            ];

            let mut oracles = Vec::new();
            for (name, alg) in algs {
                oracles = alg
                    .par_batch_search(&root, metric, queries)
                    .into_iter()
                    .map(|res| {
                        res.into_iter()
                            .max_by_key(|&(i, d)| abd_clam::utils::MaxItem(i, d))
                            .map(|(_, radius)| cakes::RnnChess(radius))
                            .unwrap()
                    })
                    .collect();

                let id = BenchmarkId::new(format!("{name}-k{k}"), multiplier);
                group.bench_function(id, |b| {
                    b.iter_with_large_drop(|| alg.par_batch_search(&root, metric, queries))
                });
            }

            let id = BenchmarkId::new(format!("rnn-oracle-k{k}"), multiplier);
            group.bench_function(id, |b| {
                b.iter_with_large_drop(|| {
                    oracles
                        .par_iter()
                        .zip(queries.par_iter())
                        .map(|(oracle, query)| oracle.par_search(&root, metric, query))
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
    for dataset in &datasets {
        let queries = dataset.read_test(&base, shuffle).unwrap();
        let queries = queries[..(max_queries.min(queries.len()))].to_vec();

        let items = dataset.read_train(&base, shuffle).unwrap();
        let metric = dataset.metric();

        let mut group = c.benchmark_group(dataset.name());
        run_group(&mut group, items, &metric, &queries);
        group.finish();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
