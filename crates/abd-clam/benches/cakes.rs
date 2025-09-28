//! Benchmarks for CAKES

#![expect(missing_docs)]

use std::usize;

use abd_clam::{
    cakes::{self, ParSearch},
    Cluster, DistanceValue,
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
    root: &Cluster<Id, I, T, A>,
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
        // let true_hits = cakes::KnnLinear(k).par_batch_search(&root, &metric, &queries);
        // let oracles = true_hits
        //     .iter()
        //     .map(|res| {
        //         res.iter()
        //             .max_by_key(|&(_, _, d)| abd_clam::utils::MaxItem((), d))
        //             .map(|&(_, _, radius)| cakes::RnnChess(radius))
        //             .unwrap()
        //     })
        //     .collect::<Vec<_>>();

        // let id = BenchmarkId::new(format!("KnnOracle(k={k})-{pruned_str}"), multiplier);
        // group.bench_function(id, |b| {
        //     b.iter_with_large_drop(|| {
        //         oracles
        //             .par_iter()
        //             .zip(queries.par_iter())
        //             .map(|(oracle, query)| oracle.par_search(&root, &metric, query))
        //             .collect::<Vec<_>>()
        //     })
        // });

        // let pred_hits = {
        //     oracles
        //         .par_iter()
        //         .zip(queries.par_iter())
        //         .map(|(oracle, query)| oracle.par_search(&root, &metric, query))
        //         .collect::<Vec<_>>()
        // };
        // let recall_stats = cakes::search_quality_stats(&true_hits, &pred_hits);
        // println!("Recall stats for oracle-k{k}-{pruned_str}, multiplier {multiplier}:");
        // for (stat_name, stat_value) in recall_stats {
        //     println!("    {stat_name}: {stat_value:.8}");
        // }

        let true_hits = cakes::KnnDfs(k).par_batch_search(&root, &metric, &queries);

        let algs = {
            let mut algs: Vec<Box<dyn ParSearch<_, _, _, _, _>>> = vec![Box::new(cakes::KnnDfs(k))];
            algs.push(Box::new(cakes::KnnBranch(k)));

            // for n in [10, 20, 50, 100, 200, 500, 1000] {
            //     algs.push(Box::new(cakes::approximate::KnnDfs(k, n, usize::MAX)));
            // }
            // for n in [1, 2, 5, 10, 20, 50, 100] {
            //     algs.push(Box::new(cakes::approximate::KnnDfs(k, usize::MAX, n * 100)));
            // }

            algs.push(Box::new(cakes::KnnBfs(k)));
            algs.push(Box::new(cakes::KnnRrnn(k)));

            algs
        };

        for alg in algs {
            let id = BenchmarkId::new(format!("{}-{pruned_str}", alg.to_string()), multiplier);
            group.bench_function(id, |b| {
                b.iter_with_large_drop(|| alg.par_batch_search(&root, &metric, &queries))
            });

            let pred_hits = alg.par_batch_search(&root, &metric, &queries);
            let recall_stats = cakes::search_quality_stats(&true_hits, &pred_hits);
            println!(
                "Search quality of {} with {pruned_str} tree and dataset multiplier {multiplier}:",
                alg.to_string()
            );
            for (stat_name, stat_value) in recall_stats {
                println!("    {stat_name}: {stat_value:.8}");
            }
        }
    }
}

/// Run benchmarks for a given dataset, doubling the dataset size each iteration until `max_items` is reached.
///
/// # Parameters
///
/// * `group`: the benchmark group to add benchmarks to
/// * `rng`: a random number generator for shuffling and data augmentation
/// * `dataset`: the dataset to benchmark
/// * `base`: the base directory where datasets are stored
/// * `max_items`: the maximum number of items to allow in the augmented dataset
/// * `max_queries`: the maximum number of queries to use from the dataset
/// * `shuffle`: whether to shuffle the dataset before building the tree
/// * `ks`: the values of k to benchmark for k-NN search
/// * `prune`: whether to also benchmark on a pruned tree based on compression costs
fn run_group<P: AsRef<std::path::Path>, R: rand::Rng>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    rng: &mut R,
    dataset: &AnnDataset,
    base: &P,
    max_items: usize,
    max_queries: usize,
    shuffle: bool,
    ks: &[usize],
    prune: bool,
) {
    let mut items = dataset
        .read_train(base, if shuffle { Some(rng) } else { None })
        .unwrap();
    let metric = dataset.metric();
    let queries = dataset.read_test(base, if shuffle { Some(rng) } else { None }).unwrap();
    let queries = queries[..(max_queries.min(queries.len()))].to_vec();

    config_group(group, queries.len());

    // Closure to compute compression costs for a ball in a post-order traversal.
    // This is a closure so it can capture `metric`.
    let compute_compression_costs = |ball: &Cluster<_, _, _, CompressionCosts<_>>| {
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
    let to_prune_or_not_to_prune = |b: &Cluster<_, _, _, _>| {
        b.annotation()
            .map_or(false, |&CompressionCosts { unitary, recursive, .. }| {
                unitary <= recursive
            })
    };

    // We will change this error value at the end of each iteration
    let mut augmentation_error = 0.0;
    for multiplier in (0..).map(|p| 2_usize.pow(p)) {
        if multiplier > 1 && items.len() * multiplier > max_items {
            // Stop if augmentation would exceed max_items
            break;
        }

        if multiplier > 1 {
            // We have had at least one iteration, so we augment the data with small random noise
            let dimensionality = items[0].len();
            let dimensional_error = augmentation_error / (dimensionality as f32).sqrt();
            let perturbations = symagen::random_data::random_tabular(
                items.len(),
                dimensionality,
                1.0 - dimensional_error,
                1.0 + dimensional_error,
                rng,
            );
            items = items
                .into_par_iter()
                .zip(perturbations)
                .flat_map(|(point, perturbation)| {
                    let perturbed_point = point.iter().zip(perturbation).map(|(&x, y)| x + y).collect();
                    [point, perturbed_point]
                })
                .collect();
        }

        // Build a tree with no annotations and benchmark the search algorithms
        let root = Cluster::par_new_tree_minimal(items, &metric, &|_| true).unwrap();
        bench_for_ks(group, &root, &metric, &queries, false, multiplier, ks);

        let root = if prune {
            // We change the annotation type to CompressionCosts and compute them in post-order, in one pass over the tree
            let post = |b: &_, _: _| Some(compute_compression_costs(b));
            let mut root = root.change_annotation_type(&post);
            // Now prune the tree based on the compression costs
            root.prune(&to_prune_or_not_to_prune);
            // Finally, we don't need the annotations anymore, so we clear them to save memory
            let root = root.remove_annotations();
            // Benchmark the search algorithms on the pruned tree
            bench_for_ks(group, &root, &metric, &queries, true, multiplier, ks);
            root
        } else {
            root
        };

        // Set the augmentation error to 0.1% of the radius of the root ball for the next iteration
        augmentation_error = root.radius() / 1000.0;
        items = root.take_all_items().into_iter().map(|(_, p)| p).collect();
    }
}

fn config_group(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>, n_queries: usize) {
    group.sample_size(10);
    group.throughput(criterion::Throughput::Elements(n_queries as u64));

    let plot_config = criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic);
    group.plot_config(plot_config);
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let datasets = [
        // For the paper
        AnnDataset::FashionMnist,
        AnnDataset::Glove25,
        AnnDataset::Sift,
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
    let mut rng = rand::rng();

    // Set these parameters to control the runtime of the benchmarks. These settings go all out and will take a long time.
    let max_items = 100_000;
    let max_queries = 100;
    let ks = [10, 100];
    let prune = false;

    let base = base_dir().unwrap();
    for dataset in &datasets[..1] {
        // For the paper, only use the first 3 datasets
        let mut group = c.benchmark_group(dataset.name());
        run_group(
            &mut group,
            &mut rng,
            dataset,
            &base,
            max_items,
            max_queries,
            shuffle,
            &ks,
            prune,
        );
        group.finish();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
