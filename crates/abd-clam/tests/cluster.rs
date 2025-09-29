//! Tests for the `Cluster` struct.

use abd_clam::{cluster::PartitionStrategy, Cluster};
use num::Integer;
use test_case::test_case;

mod common;

#[test]
fn new() -> Result<(), String> {
    let items = vec![vec![1, 2], vec![3, 4], vec![5, 6], vec![7, 8], vec![11, 12]];
    let cardinality = items.len();
    let metric = common::metrics::manhattan;

    // Don't partition in the root so we can run some tests.
    let strategy = PartitionStrategy::default().with_predicate(|_| false);
    let root = Cluster::<_, _, _, ()>::new_tree(items.into_iter().enumerate().collect(), &metric, &strategy)?;

    assert_eq!(root.cardinality(), cardinality, "Cardinality mismatch: {root:?}");
    assert!(!root.is_singleton(), "Root should not be a singleton: {root:?}");
    assert!(root.is_leaf(), "Root should be a leaf: {root:?}");
    assert_eq!(root.center(), &vec![5, 6], "Center mismatch: {root:?}");
    assert_eq!(root.radius(), 12, "Radius mismatch: {root:?}");
    assert!(root.annotation().is_none(), "Annotation should be None: {root:?}");

    let strategy = PartitionStrategy::default().with_branching_factor(2.into());
    let root = root.partition(&metric, &strategy);

    assert_eq!(root.cardinality(), cardinality, "Cardinality mismatch: {root:?}");
    assert!(!root.is_singleton(), "Root should not be a singleton: {root:?}");
    assert!(!root.is_leaf(), "Root should not be a leaf: {root:?}");

    let subtree = root.subtree();
    assert_eq!(
        subtree.len(),
        3, // Because both children will have cardinality == 2, and so will not be partitioned further
        "Subtree cardinality mismatch: [{:?}]",
        subtree
            .iter()
            .map(|c| format!("{c:?}"))
            .collect::<Vec<_>>()
            .join(" :: "),
    );

    Ok(())
}

#[test]
fn par_new() -> Result<(), String> {
    let items = vec![vec![1, 2], vec![3, 4], vec![5, 6], vec![7, 8], vec![11, 12]];
    let cardinality = items.len();
    let metric = common::metrics::manhattan;

    let strategy = PartitionStrategy::default().with_predicate(|_| false);
    let root = Cluster::<_, _, _, ()>::par_new_tree(items.into_iter().enumerate().collect(), &metric, &strategy)?;

    assert_eq!(root.cardinality(), cardinality, "Cardinality mismatch: {root:?}");
    assert!(!root.is_singleton(), "Root should not be a singleton: {root:?}");
    assert!(root.is_leaf(), "Root should be a leaf: {root:?}");
    assert_eq!(root.center(), &vec![5, 6], "Center mismatch: {root:?}");
    assert_eq!(root.radius(), 12, "Radius mismatch: {root:?}");

    let strategy = PartitionStrategy::default().with_branching_factor(2.into());
    let root = root.par_partition(&metric, &strategy);

    assert_eq!(root.cardinality(), cardinality, "Cardinality mismatch: {root:?}");
    assert!(!root.is_singleton(), "Root should not be a singleton: {root:?}");
    assert!(!root.is_leaf(), "Root should not be a leaf: {root:?}");

    let subtree = root.subtree();
    assert_eq!(
        subtree.len(),
        3, // Because both children will have cardinality == 2, and so will not be partitioned further
        "Subtree cardinality mismatch: [{:?}]",
        subtree
            .iter()
            .map(|c| format!("{c:?}"))
            .collect::<Vec<_>>()
            .join(" :: "),
    );

    Ok(())
}

#[test_case(1_000, 10 ; "1_000x10")]
#[test_case(1_000, 100 ; "1_000x100")]
#[test_case(10_000, 10 ; "10_000x10")]
#[test_case(10_000, 100 ; "10_000x100")]
fn big(car: usize, dim: usize) -> Result<(), String> {
    let metric = common::metrics::euclidean::<_, _, f32>;
    let (min, max) = (-1.0, 1.0);
    let hypot = ((4 * dim) as f32).sqrt();

    let mut ratios = Vec::new();
    for i in 0..10 {
        let data = common::data_gen::tabular(car, dim, min, max);
        let root = Cluster::par_new_tree_minimal(data, &metric)?;

        let n_clusters = root.subtree().len();

        // These bounds were derived for large `car`
        let min_ratio = 2.0 / 3.0;
        let max_ratio = 1.0;
        let ratio = n_clusters as f64 / car as f64;

        assert!(
            (min_ratio..=max_ratio).contains(&ratio),
            "Unexpected number of clusters: {n_clusters} for {car} items (ratio: {ratio:.3}, expected range: [{min_ratio}, {max_ratio}])"
        );
        ratios.push(ratio);
        assert_eq!(ratios.len(), i + 1);

        assert_eq!(root.cardinality(), car, "Cardinality mismatch: {root:?}");
        assert!(!root.is_singleton(), "Root should not be a singleton: {root:?}");
        assert!(!root.is_leaf(), "Root should not be a leaf: {root:?}");
        assert!(root.radius() <= hypot / 2.0, "Radius too large: {:.6}", root.radius());
    }

    Ok(())
}

#[test_case(1_000, 10 ; "1_000x10")]
#[test_case(10_000, 10 ; "10_000x10")]
fn par_big(car: usize, dim: usize) -> Result<(), String> {
    let metric = common::metrics::euclidean::<_, _, f32>;
    let (min, max) = (-1.0, 1.0);
    let hypot = ((4 * dim) as f32).sqrt();

    let mut ratios = Vec::new();
    for i in 0..10 {
        let data = common::data_gen::tabular(car, dim, min, max);
        let root = Cluster::par_new_tree_minimal(data, &metric)?;

        let n_clusters = root.subtree().len();

        // These bounds were derived for large `car`
        let min_ratio = 2.0 / 3.0;
        let max_ratio = 1.0;
        let ratio = n_clusters as f64 / car as f64;

        assert!(
            (min_ratio..=max_ratio).contains(&ratio),
            "Unexpected number of clusters: {n_clusters} for {car} items (ratio: {ratio:.3}, expected range: [{min_ratio}, {max_ratio}])"
        );
        ratios.push(ratio);
        assert_eq!(ratios.len(), i + 1);

        assert_eq!(root.cardinality(), car, "Cardinality mismatch: {root:?}");
        assert!(!root.is_singleton(), "Root should not be a singleton: {root:?}");
        assert!(!root.is_leaf(), "Root should not be a leaf: {root:?}");
        assert!(root.radius() <= hypot / 2.0, "Radius too large: {:.6}", root.radius());
    }

    Ok(())
}

#[test]
#[ignore = "This is just to gather some statistics about the number of clusters created in a binary tree"]
fn counting_clusters() {
    let max_n = 1_000_001;
    let mut memo = vec![0; max_n];

    memo[1] = 1;
    memo[2] = 1;
    for n in 3..max_n {
        let (q, r) = n.div_rem(&2);
        memo[n] = if r == 1 {
            1 + 2 * memo[q]
        } else {
            1 + memo[q] + memo[q - 1]
        };
    }

    let values = memo
        .into_iter()
        .enumerate()
        .skip(1023) // Skip small n where the ratio is very noisy
        .map(|(i, v)| ((i, v), v as f64 / i as f64))
        .collect::<Vec<_>>();

    let min_i = values
        .iter()
        .min_by_key(|&&((i, _), r)| abd_clam::utils::MinItem(i, r))
        .map(|&((i, _), _)| i)
        .unwrap();
    let ((min_i, min_v), min_r) = values[min_i];

    let max_i = values
        .iter()
        .max_by_key(|&&((i, _), r)| abd_clam::utils::MaxItem(i, r))
        .map(|&((i, _), _)| i)
        .unwrap();
    let ((max_i, max_v), max_r) = values[max_i];

    let mean_r = values.iter().map(|&(_, r)| r).sum::<f64>() / (values.len() as f64);
    let var_r = values.iter().map(|&(_, r)| (r - mean_r).powi(2)).sum::<f64>() / (values.len() as f64);
    let std_r = var_r.sqrt();

    let ((last_i, last_v), last_r) = values.last().copied().unwrap();

    let min_line = format!("n with min ratio: {min_i}, num clusters: {min_v}, ratio: {min_r:.6}");
    let max_line = format!("n with max ratio: {max_i}, num clusters: {max_v}, ratio: {max_r:.6}");
    let mean_line = format!("mean ratio: {mean_r:.6}, std: {std_r:.6}");
    let last_line = format!("final n = {last_i}, num clusters: {last_v}, ratio: {last_r:.6}");
    let output = format!("{min_line}\n{max_line}\n{mean_line}\n{last_line}");

    assert!(max_n == 0, "{output}"); // Always fails, just to print the output
}
