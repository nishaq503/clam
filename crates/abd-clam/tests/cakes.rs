//! Tests for the CAKES search algorithms.

use abd_clam::{
    cakes::{KnnBfs, KnnBranch, KnnDfs, KnnLinear, KnnRrnn, RnnChess, RnnLinear, Search},
    utils::MaxItem,
    DistanceValue, Tree,
};
use rayon::prelude::*;

use test_case::test_case;

mod common;

fn metric(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    let d = common::metrics::euclidean::<_, _, f32>(a, b);
    // Truncate to 3 decimal places for easier debugging
    (d * 1000.0).trunc() / 1000.0
}

#[test_case(10, 2; "10x2")]
#[test_case(1_000, 2; "1_000x2")]
#[test_case(10_000, 10; "10_000x10")]
fn vectors(car: usize, dim: usize) -> Result<(), String> {
    let data = common::data_gen::tabular(car, dim, -1.0, 1.0);
    let query = vec![0.0; dim];

    // Truncate all items to 3 decimal places for debugging
    let data = data
        .into_iter()
        .map(|v| {
            v.into_iter()
                .map(|x| (x * 1000.0).trunc() / 1000.0)
                .collect::<Vec<f32>>()
        })
        .collect::<Vec<_>>();

    println!("Starting test with {} items of dimension {}", car, dim);
    let tree = Tree::new_minimal(data.clone(), metric)?;

    let tree_items = tree
        .items_in_cluster(tree.root())
        .iter()
        .enumerate()
        .map(|(i, (id, item))| format!("  {i}: id={id}, item={item:?}"))
        .collect::<Vec<_>>();
    println!("Items: [\n{}\n]", tree_items.join("\n"));
    println!("Tree:\n{}", tree.root());

    for radius in [0.5, 1.0, 1.5, 2.0] {
        let expected_hits = tree
            .items_in_cluster(tree.root())
            .iter()
            .filter_map(|(i, item)| {
                let dist = metric(item, &query);
                if dist <= radius {
                    Some((*i, dist))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let expected_hits = sort_nondescending(expected_hits);

        let linear_alg = RnnLinear(radius);
        let linear_hits = linear_alg.search(&tree, &query);
        let linear_hits = sort_nondescending(linear_hits);
        check_hits(&expected_hits, &linear_hits, format!("RnnLinear({radius})"))?;

        let clustered_alg = RnnChess(radius);
        let clustered_hits = clustered_alg.search(&tree, &query);
        let clustered_hits = sort_nondescending(clustered_hits);
        check_hits(&expected_hits, &clustered_hits, format!("RnnChess({radius})"))?;
    }

    for k in [1, 10, 100] {
        let expected_hits = tree
            .items_in_cluster(tree.root())
            .iter()
            .map(|(i, item)| (*i, metric(item, &query)))
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
        let linear_hits = linear_alg.search(&tree, &query);
        let linear_hits = sort_nondescending(linear_hits);
        check_hits(&expected_hits, &linear_hits, format!("KnnLinear({k})"))?;

        let dfs_alg = KnnDfs(k);
        let dfs_hits = dfs_alg.search(&tree, &query);
        let dfs_hits = sort_nondescending(dfs_hits);
        check_hits(&expected_hits, &dfs_hits, format!("KnnDfs({k})"))?;

        let branch_alg = KnnBranch(k);
        let branch_hits = branch_alg.search(&tree, &query);
        let branch_hits = sort_nondescending(branch_hits);
        check_hits(&expected_hits, &branch_hits, format!("KnnBranch({k})"))?;

        let bfs_alg = KnnBfs(k);
        let bfs_hits = bfs_alg.search(&tree, &query);
        let bfs_hits = sort_nondescending(bfs_hits);
        check_hits(&expected_hits, &bfs_hits, format!("KnnBfs({k})"))?;

        let rrnn_alg = KnnRrnn(k);
        let rrnn_hits = rrnn_alg.search(&tree, &query);
        let rrnn_hits = sort_nondescending(rrnn_hits);
        check_hits(&expected_hits, &rrnn_hits, format!("KnnRrnn({k})"))?;
    }

    Ok(())
}

#[test_case(10_000, 10; "10_000x10")]
#[test_case(10_000, 100; "10_000x100")]
#[test_case(100_000, 10; "100_000x10")]
#[test_case(100_000, 100; "100_000x100")]
fn par_vectors(car: usize, dim: usize) -> Result<(), String> {
    let data = common::data_gen::tabular(car, dim, -1.0, 1.0);
    let query = vec![0.0; dim];

    let tree = Tree::par_new_minimal(data.clone(), metric)?;

    for radius in [1.0, 1.5, 2.0] {
        let expected_hits = tree
            .items_in_cluster(tree.root())
            .par_iter()
            .filter_map(|(i, item)| {
                let dist = metric(item, &query);
                if dist <= radius {
                    Some((*i, dist))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let expected_hits = sort_nondescending(expected_hits);

        let linear_alg = RnnLinear(radius);
        let linear_hits = linear_alg.par_search(&tree, &query);
        let linear_hits = sort_nondescending(linear_hits);
        check_hits(&expected_hits, &linear_hits, format!("RnnLinear({radius})"))?;

        let clustered_alg = RnnChess(radius);
        let clustered_hits = clustered_alg.par_search(&tree, &query);
        let clustered_hits = sort_nondescending(clustered_hits);
        check_hits(&expected_hits, &clustered_hits, format!("RnnChess({radius})"))?;
    }

    for k in [1, 10, 100] {
        let expected_hits = tree
            .items_in_cluster(tree.root())
            .par_iter()
            .map(|(i, item)| (*i, metric(item, &query)))
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
        let linear_hits = linear_alg.par_search(&tree, &query);
        let linear_hits = sort_nondescending(linear_hits);
        check_hits(&expected_hits, &linear_hits, format!("KnnLinear({k})"))?;

        let dfs_alg = KnnDfs(k);
        let dfs_hits = dfs_alg.par_search(&tree, &query);
        let dfs_hits = sort_nondescending(dfs_hits);
        check_hits(&expected_hits, &dfs_hits, format!("KnnDfs({k})"))?;

        let branch_alg = KnnBranch(k);
        let branch_hits = branch_alg.par_search(&tree, &query);
        let branch_hits = sort_nondescending(branch_hits);
        check_hits(&expected_hits, &branch_hits, format!("KnnBranch({k})"))?;

        let bfs_alg = KnnBfs(k);
        let bfs_hits = bfs_alg.par_search(&tree, &query);
        let bfs_hits = sort_nondescending(bfs_hits);
        check_hits(&expected_hits, &bfs_hits, format!("KnnBfs({k})"))?;

        let rrnn_alg = KnnRrnn(k);
        let rrnn_hits = rrnn_alg.par_search(&tree, &query);
        let rrnn_hits = sort_nondescending(rrnn_hits);
        check_hits(&expected_hits, &rrnn_hits, format!("KnnRrnn({k})"))?;
    }

    Ok(())
}

fn check_hits<T: DistanceValue>(expected: &[(usize, T)], actual: &[(usize, T)], alg_name: String) -> Result<(), String> {
    assert_eq!(
        expected.len(),
        actual.len(),
        "{alg_name}: Hit count mismatch: \nexp {expected:?}, \ngot {actual:?}",
    );

    for (i, (&(_, e), &(_, a))) in expected.iter().zip(actual.iter()).enumerate() {
        assert_eq!(
            e, a,
            "{alg_name}: Distance mismatch at index {i}: \nexp {expected:?}, \ngot {actual:?}",
        );
    }

    Ok(())
}

fn sort_nondescending<T: PartialOrd>(mut items: Vec<(usize, T)>) -> Vec<(usize, T)> {
    items.sort_by(|(_, l), (_, r)| MaxItem((), l).cmp(&MaxItem((), r)));
    items
}
