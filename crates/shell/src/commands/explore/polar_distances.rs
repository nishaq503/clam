//! Save distances from polar points to all other points in each `Cluster`.

use std::path::Path;

use abd_clam::{Cluster, DistanceValue, Tree};

use crate::trees::{ShellTree, VectorTree};

/// Save distances from polar points to all other points in each `Cluster`.
pub fn polar_distances<P: AsRef<Path>>(tree: ShellTree, out_dir: P) -> Result<(), String> {
    match tree {
        ShellTree::Lcs(tree) => save_cluster_distances(tree, out_dir),
        ShellTree::Levenshtein(tree) => save_cluster_distances(tree, out_dir),
        ShellTree::Euclidean(tree) => match tree {
            VectorTree::F32(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::F64(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::I8(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::I16(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::I32(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::I64(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::U8(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::U16(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::U32(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::U64(tree) => save_cluster_distances(tree, out_dir),
        },
        ShellTree::Cosine(tree) => match tree {
            VectorTree::F32(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::F64(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::I8(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::I16(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::I32(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::I64(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::U8(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::U16(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::U32(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::U64(tree) => save_cluster_distances(tree, out_dir),
        },
    }
}

/// Save distances from polar points to all other points in each `Cluster`.
fn save_cluster_distances<Id, I, T, A, M, P>(tree: Tree<Id, I, T, A, M>, out_dir: P) -> Result<(), String>
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + serde::Serialize + Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
    P: AsRef<Path>,
{
    let (items, root, metric) = tree.deconstruct();
    let distance_matrix = abd_clam::utils::par_pairwise_distances(&items, &metric);

    for (mut cluster, span) in root.clear_annotations().unstacked_postorder_owned() {
        if let Some(s) = span {
            // We will only annotate parent clusters.
            cluster.annotate((s, filter_polar_distances(&distance_matrix, &cluster)))
        };
        let out_path = out_dir.as_ref().join(format!("cluster_{}.json", cluster.center_index()));
        let contents = serde_json::to_string_pretty(&cluster).map_err(|e| format!("Failed to serialize cluster: {e}"))?;
        std::fs::write(&out_path, contents).map_err(|e| format!("Failed to write cluster to {out_path:?}: {e}"))?;
    }

    Ok(())
}

/// Get distances from polar points to all other points in the `Cluster`.
///
/// The `polar points` are defined as follows:
///   - The `left polar point` is the point in the cluster that is farthest from the cluster center.
///   - The `right polar point` is the point in the cluster that is farthest from the left polar point.
fn filter_polar_distances<T, A>(distance_matrix: &[Vec<T>], cluster: &Cluster<T, A>) -> Vec<(T, T)>
where
    T: DistanceValue + Send + Sync,
{
    let center_distances = &distance_matrix[cluster.center_index()][cluster.all_items_indices()];
    let arg_left = {
        let (arg_max, _) = center_distances
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or_else(|| unreachable!("Cluster has no items"));
        arg_max + cluster.center_index()
    };
    let left_distances = &distance_matrix[arg_left][cluster.all_items_indices()];
    let arg_right = {
        let (arg_max, _) = center_distances
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or_else(|| unreachable!("Cluster has no items"));
        arg_max + cluster.center_index()
    };
    let right_distances = &distance_matrix[arg_right][cluster.all_items_indices()];
    left_distances.iter().copied().zip(right_distances.iter().copied()).collect()
}
