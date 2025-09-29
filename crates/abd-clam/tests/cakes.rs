//! Tests for the CAKES search algorithms.

use std::fmt::Debug;

use abd_clam::{
    cakes::{KnnBfs, KnnBranch, KnnDfs, KnnLinear, KnnRrnn, RnnChess, RnnLinear, Search},
    utils::MaxItem,
    Cluster, DistanceValue,
};
use rayon::prelude::*;

use test_case::test_case;

mod common;

#[test_case(10, 2; "10 x 2")]
#[test_case(1_000, 2; "1_000 x 2")]
#[test_case(10_000, 10; "10_000 x 10")]
fn vectors(car: usize, dim: usize) -> Result<(), String> {
    let data = common::data_gen::tabular(car, dim, -1.0, 1.0);
    let metric = common::metrics::euclidean::<_, _, f32>;
    let query = vec![0.0; dim];

    let root = Cluster::new_tree_minimal(data.clone(), &metric)?;
    let all_items = root.all_items();

    for radius in [1.0, 1.5, 2.0] {
        let expected_hits = all_items
            .iter()
            .filter_map(|(i, item)| {
                let dist = metric(item, &query);
                if dist <= radius {
                    Some((i, item, dist))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let expected_hits = sort_nondescending(expected_hits);

        let linear_alg = RnnLinear(radius);
        let linear_hits = linear_alg.search(&root, &metric, &query);
        let linear_hits = sort_nondescending(linear_hits);
        check_hits(&expected_hits, &linear_hits, format!("RnnLinear({radius})"))?;

        let clustered_alg = RnnChess(radius);
        let clustered_hits = clustered_alg.search(&root, &metric, &query);
        let clustered_hits = sort_nondescending(clustered_hits);
        check_hits(&expected_hits, &clustered_hits, format!("RnnChess({radius})"))?;
    }

    for k in [1, 10, 100] {
        let expected_hits = all_items
            .iter()
            .map(|(i, item)| (i, item, metric(item, &query)))
            .collect::<Vec<_>>();
        let expected_hits = sort_nondescending(expected_hits)
            .into_iter()
            .take(k)
            .collect::<Vec<_>>();

        assert_eq!(
            expected_hits.len(),
            k.min(car),
            "Not enough expected hits {} for k={}",
            expected_hits.len(),
            k.min(car)
        );

        let linear_alg = KnnLinear(k);
        let linear_hits = linear_alg.search(&root, &metric, &query);
        let linear_hits = sort_nondescending(linear_hits);
        check_hits(&expected_hits, &linear_hits, format!("KnnLinear({k})"))?;

        let dfs_alg = KnnDfs(k);
        let dfs_hits = dfs_alg.search(&root, &metric, &query);
        let dfs_hits = sort_nondescending(dfs_hits);
        check_hits(&expected_hits, &dfs_hits, format!("KnnDfs({k})"))?;

        let branch_alg = KnnBranch(k);
        let branch_hits = branch_alg.search(&root, &metric, &query);
        let branch_hits = sort_nondescending(branch_hits);
        check_hits(&expected_hits, &branch_hits, format!("KnnBranch({k})"))?;

        let bfs_alg = KnnBfs(k);
        let bfs_hits = bfs_alg.search(&root, &metric, &query);
        let bfs_hits = sort_nondescending(bfs_hits);
        check_hits(&expected_hits, &bfs_hits, format!("KnnBfs({k})"))?;

        let rrnn_alg = KnnRrnn(k);
        let rrnn_hits = rrnn_alg.search(&root, &metric, &query);
        let rrnn_hits = sort_nondescending(rrnn_hits);
        check_hits(&expected_hits, &rrnn_hits, format!("KnnRrnn({k})"))?;
    }

    Ok(())
}

#[test_case(10_000, 10; "10_000 x 10")]
#[test_case(10_000, 100; "10_000 x 100")]
#[test_case(100_000, 10; "100_000 x 10")]
#[test_case(100_000, 100; "100_000 x 100")]
fn par_vectors(car: usize, dim: usize) -> Result<(), String> {
    let data = common::data_gen::tabular(car, dim, -1.0, 1.0);
    let metric = common::metrics::euclidean::<_, _, f32>;
    let query = vec![0.0; dim];

    let root = Cluster::par_new_tree_minimal(data.clone(), &metric)?;
    let all_items = root.all_items();

    for radius in [1.0, 1.5, 2.0] {
        let expected_hits = all_items
            .par_iter()
            .filter_map(|(i, item)| {
                let dist = metric(item, &query);
                if dist <= radius {
                    Some((i, item, dist))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let expected_hits = sort_nondescending(expected_hits);

        let linear_alg = RnnLinear(radius);
        let linear_hits = linear_alg.par_search(&root, &metric, &query);
        let linear_hits = sort_nondescending(linear_hits);
        check_hits(&expected_hits, &linear_hits, format!("RnnLinear({radius})"))?;

        let clustered_alg = RnnChess(radius);
        let clustered_hits = clustered_alg.par_search(&root, &metric, &query);
        let clustered_hits = sort_nondescending(clustered_hits);
        check_hits(&expected_hits, &clustered_hits, format!("RnnChess({radius})"))?;
    }

    for k in [1, 10, 100] {
        let expected_hits = all_items
            .par_iter()
            .map(|(i, item)| (i, item, metric(item, &query)))
            .collect::<Vec<_>>();
        let expected_hits = sort_nondescending(expected_hits)
            .into_iter()
            .take(k)
            .collect::<Vec<_>>();

        assert_eq!(
            expected_hits.len(),
            k.min(car),
            "Not enough expected hits {} for k={}",
            expected_hits.len(),
            k.min(car)
        );

        let linear_alg = KnnLinear(k);
        let linear_hits = linear_alg.par_search(&root, &metric, &query);
        let linear_hits = sort_nondescending(linear_hits);
        check_hits(&expected_hits, &linear_hits, format!("KnnLinear({k})"))?;

        let dfs_alg = KnnDfs(k);
        let dfs_hits = dfs_alg.par_search(&root, &metric, &query);
        let dfs_hits = sort_nondescending(dfs_hits);
        check_hits(&expected_hits, &dfs_hits, format!("KnnDfs({k})"))?;

        let branch_alg = KnnBranch(k);
        let branch_hits = branch_alg.par_search(&root, &metric, &query);
        let branch_hits = sort_nondescending(branch_hits);
        check_hits(&expected_hits, &branch_hits, format!("KnnBranch({k})"))?;

        let bfs_alg = KnnBfs(k);
        let bfs_hits = bfs_alg.par_search(&root, &metric, &query);
        let bfs_hits = sort_nondescending(bfs_hits);
        check_hits(&expected_hits, &bfs_hits, format!("KnnBfs({k})"))?;

        let rrnn_alg = KnnRrnn(k);
        let rrnn_hits = rrnn_alg.par_search(&root, &metric, &query);
        let rrnn_hits = sort_nondescending(rrnn_hits);
        check_hits(&expected_hits, &rrnn_hits, format!("KnnRrnn({k})"))?;
    }

    Ok(())
}

fn check_hits<Id: Debug, I: Debug, T: DistanceValue + Debug>(
    expected: &[(&Id, &I, T)],
    actual: &[(&Id, &I, T)],
    alg_name: String,
) -> Result<(), String> {
    assert_eq!(
        expected.len(),
        actual.len(),
        "{alg_name}: Hit count mismatch: \nexp {:?}, \ngot {:?}",
        expected.iter().map(|(_, _, d)| d).collect::<Vec<_>>(),
        actual.iter().map(|(_, _, d)| d).collect::<Vec<_>>()
    );

    for (i, (&(_, _, e), &(_, _, a))) in expected.iter().zip(actual.iter()).enumerate() {
        assert_eq!(
            e,
            a,
            "{alg_name}: Distance mismatch at index {i}: \nexp {:?}, \ngot {:?}",
            expected.iter().map(|(_, _, d)| d).collect::<Vec<_>>(),
            actual.iter().map(|(_, _, d)| d).collect::<Vec<_>>()
        );
    }

    Ok(())
}

fn sort_nondescending<'a, Id, I, T: PartialOrd + Copy>(mut items: Vec<(&'a Id, &'a I, T)>) -> Vec<(&'a Id, &'a I, T)> {
    items.sort_by_key(|&(_, _, d)| MaxItem((), d));
    items
}
