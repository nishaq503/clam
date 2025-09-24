//! Benchmarks for CAKES

#![expect(missing_docs)]

use abd_clam::{
    cakes::{self, ParSearch},
    Ball, DistanceValue,
};
use criterion::{criterion_group, criterion_main, Criterion};

mod utils;

use utils::{
    ann_benchmarks::{base_dir, AnnDataset},
    metrics,
};

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
    for k in [10] {
        let knn_linear = cakes::KnnLinear(k);
        let id = format!("knn-linear-{k}");
        group.bench_function(&id, |b| {
            b.iter_with_large_drop(|| knn_linear.par_batch_search(root, metric, queries));
        });

        let knn_dfs = cakes::KnnDfs(k);
        let id = format!("knn-dfs-{k}");
        group.bench_function(&id, |b| {
            b.iter_with_large_drop(|| knn_dfs.par_batch_search(root, metric, queries));
        });

        let knn_bfs = cakes::KnnBfs(k);
        let id = format!("knn-bfs-{k}");
        group.bench_function(&id, |b| {
            b.iter_with_large_drop(|| knn_bfs.par_batch_search(root, metric, queries));
        });

        // let knn_rrnn = cakes::KnnRrnn(k);
        // let id = format!("knn-rrnn-{k}");
        // group.bench_function(&id, |b| {
        //     b.iter_with_large_drop(|| knn_rrnn.par_batch_search(root, metric, queries));
        // });
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let euc_datasets = [
        AnnDataset::FashionMnist,
        AnnDataset::Mnist,
        AnnDataset::Sift,
        AnnDataset::Gist,
    ];
    let cos_datasets = [
        AnnDataset::Glove25,
        AnnDataset::Glove50,
        AnnDataset::Glove100,
        AnnDataset::Glove200,
        AnnDataset::DeepImage,
    ];

    let shuffle = true;
    let max_queries = 100;

    let base = base_dir().unwrap();
    for dataset in &euc_datasets {
        let items = dataset.read_train(&base, shuffle).unwrap();
        let queries = dataset.read_test(&base, shuffle).unwrap()[..max_queries].to_vec();
        let metric = metrics::euclidean;

        let root = Ball::par_new_tree(items, &metric, &|_| true).unwrap();
        let mut group = c.benchmark_group(format!("euc-{}", dataset.name()));
        group
            .throughput(criterion::Throughput::Elements(queries.len() as u64))
            .sample_size(10);
        run_group(&mut group, &root, &metric, &queries);
        group.finish();
    }

    for dataset in &cos_datasets {
        let items = dataset.read_train(&base, shuffle).unwrap();
        let queries = dataset.read_test(&base, shuffle).unwrap()[..max_queries].to_vec();
        let metric = metrics::cosine;

        let root = Ball::par_new_tree(items, &metric, &|_| true).unwrap();

        let mut group = c.benchmark_group(format!("cos-{}", dataset.name()));
        group
            .throughput(criterion::Throughput::Elements(queries.len() as u64))
            .sample_size(10);
        run_group(&mut group, &root, &metric, &queries);
        group.finish();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
