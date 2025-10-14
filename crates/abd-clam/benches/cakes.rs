//! Benchmarks for CAKES

#![expect(missing_docs)]

use std::usize;

use rand::prelude::*;

use abd_clam::{
    cakes::{self, BatchedSearch},
    cluster::{BranchingFactor, PartitionStrategy},
    tree, Cluster, DistanceValue,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rayon::prelude::*;

mod utils;

use utils::ann_benchmarks::{base_dir, AnnDataset};

fn bench_for_ks<Id, I, T, A, M>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    tree: &tree::Tree<Id, I, T, A, M>,
    root: &Cluster<Id, I, T, A>,
    queries: &[I],
    multiplier: usize,
    ks: &[usize],
) where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
{
    for &k in ks {
        let true_hits = cakes::KnnLinear(k).par_batch_search(&root, tree.metric(), &queries);

        // Benchmark the exact algorithms
        bench_one_alg(
            group,
            (tree, &tree::cakes::KnnDfs(k)),
            (root, &cakes::KnnDfs(k)),
            queries,
            &true_hits,
            multiplier,
        );
        bench_one_alg(
            group,
            (tree, &tree::cakes::KnnBfs(k)),
            (root, &cakes::KnnBfs(k)),
            queries,
            &true_hits,
            multiplier,
        );
        bench_one_alg(
            group,
            (tree, &tree::cakes::KnnBranch(k)),
            (root, &cakes::KnnBranch(k)),
            queries,
            &true_hits,
            multiplier,
        );
        bench_one_alg(
            group,
            (tree, &tree::cakes::KnnRrnn(k)),
            (root, &cakes::KnnRrnn(k)),
            queries,
            &true_hits,
            multiplier,
        );

        // // Benchmark the approximate algorithms
        // for n in [10, 100, 1000] {
        //     // Varying number of leaves explored
        //     bench_one_alg(
        //         group,
        //         &cakes::approximate::KnnDfs(k, n, usize::MAX),
        //         root,
        //         metric,
        //         queries,
        //         &true_hits,
        //         pruned_str,
        //         multiplier,
        //     );
        //     // Varying number of distance computations performed
        //     bench_one_alg(
        //         group,
        //         &cakes::approximate::KnnDfs(k, usize::MAX, n * 100),
        //         root,
        //         metric,
        //         queries,
        //         &true_hits,
        //         pruned_str,
        //         multiplier,
        //     );
        // }
    }
}

fn bench_one_alg<Id, I, T, A, M, Alg, TreeAlg>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    (tree, t_alg): (&tree::Tree<Id, I, T, A, M>, &TreeAlg),
    (root, r_alg): (&Cluster<Id, I, T, A>, &Alg),
    queries: &[I],
    true_hits: &[Vec<(&Id, &I, T)>],
    multiplier: usize,
) where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
    Alg: BatchedSearch<Id, I, T, A, M> + Send + Sync,
    TreeAlg: tree::cakes::Search<Id, I, T, A, M> + Send + Sync,
{
    let r_id = BenchmarkId::new(format!("Root-{}", r_alg.to_string()), multiplier);
    group.bench_function(r_id, |b| {
        b.iter_with_large_drop(|| r_alg.par_batch_search(&root, tree.metric(), &queries))
    });

    let t_id = BenchmarkId::new(format!("Tree-{}", t_alg.to_string()), multiplier);
    group.bench_function(t_id, |b| {
        b.iter_with_large_drop(|| t_alg.par_batch_search(&tree, &queries))
    });

    let all_clusters = root.subtree();
    let size_of_tree = all_clusters.len();
    let max_depth = all_clusters.iter().map(|c| c.depth()).max().unwrap_or(0);

    let all_leaves = all_clusters.iter().filter(|c| c.is_leaf()).copied().collect::<Vec<_>>();
    let leaf_fraction = all_leaves.len() as f64 / size_of_tree as f64;
    let mean_leaf_cardinality =
        all_leaves.iter().map(|c| c.cardinality()).sum::<usize>() as f64 / all_leaves.len() as f64;

    let singleton_fraction = all_leaves.iter().filter(|c| c.is_singleton()).count() as f64 / all_leaves.len() as f64;

    println!(
        "Tree stats for dataset with cardinality {} after multiplier {multiplier}:",
        root.cardinality()
    );
    println!(
        "    Number of clusters: {size_of_tree}, Ratio: {:.8}",
        size_of_tree as f64 / root.cardinality() as f64
    );
    println!("    Max depth: {max_depth}");
    println!("    Leaf fraction of clusters: {leaf_fraction:.8}, mean leaf cardinality: {mean_leaf_cardinality:.8}");
    println!("    Singleton fraction of leaves: {singleton_fraction:.8}");

    let pred_hits = r_alg.par_batch_search(&root, tree.metric(), &queries);
    let recall_stats = cakes::search_quality_stats(true_hits, &pred_hits);
    println!("Search quality of {}:", r_alg.to_string());
    for (stat_name, stat_value) in recall_stats {
        println!("    {stat_name}: {stat_value:.8}");
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
#[expect(unused_variables)]
fn run_group<P: AsRef<std::path::Path>, R: rand::Rng>(
    c: &mut Criterion,
    rng: &mut R,
    dataset: &AnnDataset,
    base: &P,
    max_items: usize,
    max_queries: usize,
    shuffle: bool,
    branching_factors: &[usize],
    ks: &[usize],
) {
    let mut items = dataset
        .read_train(base, if shuffle { Some(rng) } else { None })
        .unwrap();
    let metric = dataset.metric();
    let queries = dataset.read_test(base, if shuffle { Some(rng) } else { None }).unwrap();
    let queries = queries[..(max_queries.min(queries.len()))].to_vec();

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

        // let strategies = {
        //     let mut strategies = vec![];

        //     for &bf in branching_factors {
        //         strategies.push(PartitionStrategy::default().with_branching_factor(BranchingFactor::Fixed(bf)));
        //     }
        //     for &bf in branching_factors {
        //         if bf > 2 {
        //             strategies.push(PartitionStrategy::default().with_branching_factor(BranchingFactor::Adaptive(bf)));
        //         }
        //     }
        //     strategies.push(PartitionStrategy::default().with_branching_factor(BranchingFactor::Logarithmic));

        //     strategies.push(PartitionStrategy::default().with_span_reduction(SpanReductionFactor::Sqrt2));
        //     strategies.push(PartitionStrategy::default().with_span_reduction(SpanReductionFactor::Two));
        //     strategies.push(PartitionStrategy::default().with_span_reduction(SpanReductionFactor::E));
        //     strategies.push(PartitionStrategy::default().with_span_reduction(SpanReductionFactor::Pi));
        //     strategies.push(PartitionStrategy::default().with_span_reduction(SpanReductionFactor::Phi));

        //     strategies
        // };
        let strategies = vec![PartitionStrategy::default().with_branching_factor(BranchingFactor::Fixed(2))];

        for strategy in &strategies {
            let mut group = c.benchmark_group(format!("CAKES-{}-{strategy}", dataset.name()));
            config_group(&mut group, queries.len());

            // Build a tree with no annotations and benchmark the search algorithms
            if shuffle {
                items.shuffle(rng);
            }
            let indexed_items = items.into_iter().enumerate().collect::<Vec<_>>();

            println!("Building Cluster with strategy {strategy}");
            let root_start = std::time::Instant::now();
            let root = Cluster::<_, _, _, ()>::par_new_tree(indexed_items.clone(), &metric, strategy).unwrap();
            let root_time = root_start.elapsed();
            println!("Built Cluster in {:.6}", root_time.as_secs_f32());

            println!("Building Tree");
            let tree_start = std::time::Instant::now();
            let tree = tree::Tree::new(indexed_items, metric).unwrap();
            let tree_time = tree_start.elapsed();
            println!("Built Tree in {:.6}", tree_time.as_secs_f32());

            bench_for_ks(&mut group, &tree, &root, &queries, multiplier, ks);

            // Set the augmentation error to 0.1% of the radius of the root ball for the next augmentation
            augmentation_error = root.radius() / 1000.0;
            items = root.take_all_items().into_iter().map(|(_, p)| p).collect();

            group.finish();
        }
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
    let shuffle = false;
    let mut rng = rand::rng();

    // Set these parameters to control the runtime of the benchmarks. These settings go all out and will take a long time.
    let max_items = 100_000;
    let max_queries = 100;
    let branching_factors = [2, 10, 100];
    let ks = [10];

    let base = base_dir().unwrap();
    for dataset in &datasets {
        if !matches!(dataset, AnnDataset::FashionMnist) {
            continue; // Targeting dataset for hyperparameter tuning
        }
        // For the paper, only use the first 3 datasets
        run_group(
            c,
            &mut rng,
            dataset,
            &base,
            max_items,
            max_queries,
            shuffle,
            &branching_factors,
            &ks,
        );
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
