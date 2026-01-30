//! A multiple sequence alignment (MSA) of sequences of type `S`.

use rayon::prelude::*;

use crate::{Cluster, DistanceValue, Tree};

mod alignment_ops;
mod columnar;
mod cost_matrix;
mod sequence;

pub use alignment_ops::Direction;
pub use cost_matrix::CostMatrix;
pub use sequence::Sequence;

use alignment_ops::{Edit, Edits};
use columnar::Columnar;

/// A multiple sequence alignment (MSA) of sequences of type `S`.
#[must_use]
pub struct Msa<S: Sequence>(Vec<S>);

impl<S: Sequence> core::ops::Deref for Msa<S> {
    type Target = Vec<S>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<S: Sequence> IntoIterator for Msa<S> {
    type Item = S;
    type IntoIter = std::vec::IntoIter<S>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<S: Sequence + Send + Sync> IntoParallelIterator for Msa<S> {
    type Item = S;
    type Iter = rayon::vec::IntoIter<S>;

    fn into_par_iter(self) -> Self::Iter {
        self.0.into_par_iter()
    }
}

impl<S: Sequence> Msa<S> {
    /// Creates a new MSA from a vector of sequences.
    pub fn from_tree<Id, T, A, M>(tree: &Tree<Id, S, T, A, M>, cost_matrix: &CostMatrix<T>) -> Self
    where
        T: DistanceValue,
    {
        ftlog::info!("Creating MSA from tree with {} sequences", tree.cardinality());

        let mut root = tree.root.clone_without_annotations();
        let columnar = Self::from_cluster_collapse(&mut root, &tree.items, cost_matrix);
        ftlog::info!("Finished creating Columnar MSA from tree with {} columns", columnar.len());

        let msa = columnar.into_rows(true);
        ftlog::info!("Converted Columnar MSA to {} sequences", msa.len());

        Self(msa)
    }

    /// Collapses a Cluster into an MSA.
    fn from_cluster_collapse<Id, T>(cluster: &mut Cluster<T, Columnar<S>>, all_items: &[(Id, S)], cost_matrix: &CostMatrix<T>) -> Columnar<S>
    where
        T: DistanceValue,
    {
        let stride = 128; // Arbitrary stride over depth to avoid deep recursion

        let target_depth = cluster.depth + stride;
        let selector = |c: &Cluster<_, _>, (): &()| c.depth == target_depth;
        let targets = cluster.filter_clusters_mut(&selector, &());
        for c in targets {
            let msa = Self::from_cluster_collapse(c, all_items, cost_matrix);
            ftlog::info!(
                "Collapsing cluster at depth {} with {} sequences to MSA with {} columns and {} rows",
                c.depth,
                c.cardinality,
                msa.len(),
                msa.n_seq()
            );
            c.children = None;
            c.annotation = msa;
        }

        Self::from_cluster(cluster, all_items, cost_matrix)
    }

    /// Recursively creates an MSA from a Cluster.
    fn from_cluster<Id, T>(cluster: &mut Cluster<T, Columnar<S>>, all_items: &[(Id, S)], cost_matrix: &CostMatrix<T>) -> Columnar<S>
    where
        T: DistanceValue,
    {
        if let Some((children, _, _)) = &mut cluster.children {
            ftlog::debug!("Aligning parent cluster at depth {} with {} sequences", cluster.depth, cluster.cardinality);

            let mut children = children
                .iter_mut()
                .map(|child| Self::from_cluster(child, all_items, cost_matrix))
                .collect::<Vec<_>>();

            let mut bottom = children.pop().unwrap_or_else(|| unreachable!("Parent cluster always has children"));
            while let Some(prev) = children.pop() {
                bottom = bottom.merge(prev, cost_matrix);
            }

            let bottom = bottom.post_pend_row(&all_items[cluster.center_index].1, cost_matrix);

            ftlog::debug!(
                "Finished aligning parent cluster at depth {} with {} sequences to {} width",
                cluster.depth,
                cluster.cardinality,
                bottom.len()
            );

            bottom
        } else if cluster.annotation.is_empty() {
            let child_msa = cluster.take_annotation();
            // Cluster was previously collapsed and then turned back into a leaf.
            ftlog::debug!(
                "Aligning leaf collapsed cluster at depth {} with {} sequences",
                cluster.depth,
                cluster.cardinality
            );
            child_msa.post_pend_row(&all_items[cluster.center_index].1, cost_matrix)
        } else {
            ftlog::debug!("Aligning leaf cluster at depth {} with {} sequences", cluster.depth, cluster.cardinality);

            let mut items = all_items[cluster.all_items_indices()]
                .iter()
                .map(|(_, seq)| Columnar::from_row(seq))
                .collect::<Vec<_>>();

            let mut bottom = items.pop().unwrap_or_else(|| unreachable!("Leaf cluster is never empty"));
            while let Some(prev) = items.pop() {
                bottom = bottom.merge(prev, cost_matrix);
            }

            ftlog::debug!(
                "Finished aligning leaf cluster at depth {} with {} sequences to {} width",
                cluster.depth,
                cluster.cardinality,
                bottom.len()
            );

            bottom
        }
    }
}

/// Parallel implementations of MSA methods.
impl<S: Sequence + Send + Sync> Msa<S> {
    /// Parallel version of [`Self::from_tree`].
    pub fn par_from_tree<Id, T, A, M>(tree: &Tree<Id, S, T, A, M>, cost_matrix: &CostMatrix<T>) -> Self
    where
        Id: Send + Sync,
        T: DistanceValue + Send + Sync,
        A: Send + Sync,
        M: Send + Sync,
    {
        ftlog::info!("Creating MSA from tree with {} sequences in parallel", tree.cardinality());

        let mut root = tree.root.clone_without_annotations();
        let columnar = Self::par_from_cluster_collapse(&mut root, &tree.items, cost_matrix);
        ftlog::info!("Finished creating Columnar MSA from tree with {} columns in parallel", columnar.len());

        let msa = columnar.par_into_rows(true);
        ftlog::info!("Converted Columnar MSA to {} sequences in parallel", msa.len());

        Self(msa)
    }

    /// Parallel version of [`Self::from_cluster_collapse`].
    fn par_from_cluster_collapse<Id, T>(cluster: &mut Cluster<T, Columnar<S>>, all_items: &[(Id, S)], cost_matrix: &CostMatrix<T>) -> Columnar<S>
    where
        Id: Send + Sync,
        T: DistanceValue + Send + Sync,
    {
        let stride = 128; // Arbitrary stride over depth to avoid deep recursion

        let target_depth = cluster.depth + stride;
        let selector = |c: &Cluster<_, _>, (): &()| c.depth == target_depth;
        let targets = cluster.filter_clusters_mut(&selector, &());
        targets.into_par_iter().for_each(|c| {
            let msa = Self::par_from_cluster_collapse(c, all_items, cost_matrix);
            ftlog::info!(
                "Collapsing cluster at depth {} with {} sequences to MSA with {} columns and {} rows in parallel",
                c.depth,
                c.cardinality,
                msa.len(),
                msa.n_seq()
            );
            c.children = None;
            c.annotation = msa;
        });

        Self::par_from_cluster(cluster, all_items, cost_matrix)
    }

    /// Parallel version of [`Self::from_cluster`].
    fn par_from_cluster<Id, T>(cluster: &mut Cluster<T, Columnar<S>>, all_items: &[(Id, S)], cost_matrix: &CostMatrix<T>) -> Columnar<S>
    where
        Id: Send + Sync,
        T: DistanceValue + Send + Sync,
    {
        if let Some((children, _, _)) = &mut cluster.children {
            ftlog::debug!(
                "Aligning parent cluster at depth {} with {} sequences in parallel",
                cluster.depth,
                cluster.cardinality
            );

            let mut children = children
                .par_iter_mut()
                .map(|child| Self::par_from_cluster(child, all_items, cost_matrix))
                .collect::<Vec<_>>();

            let mut bottom = children.pop().unwrap_or_else(|| unreachable!("Parent cluster always has children"));
            while let Some(prev) = children.pop() {
                bottom = bottom.par_merge(prev, cost_matrix);
            }

            let bottom = bottom.post_pend_row(&all_items[cluster.center_index].1, cost_matrix);

            ftlog::debug!(
                "Finished aligning parent cluster at depth {} with {} sequences to {} width in parallel",
                cluster.depth,
                cluster.cardinality,
                bottom.len()
            );

            bottom
        } else if cluster.annotation.is_empty() {
            let child_msa = cluster.take_annotation();

            // Cluster was previously collapsed and then turned back into a leaf.
            ftlog::debug!(
                "Aligning leaf collapsed cluster at depth {} with {} sequences",
                cluster.depth,
                cluster.cardinality
            );
            child_msa.post_pend_row(&all_items[cluster.center_index].1, cost_matrix)
        } else {
            ftlog::debug!(
                "Aligning leaf cluster at depth {} with {} sequences in parallel",
                cluster.depth,
                cluster.cardinality
            );

            let mut items = all_items[cluster.all_items_indices()]
                .par_iter()
                .map(|(_, seq)| Columnar::from_row(seq))
                .collect::<Vec<_>>();

            let mut bottom = items.pop().unwrap_or_else(|| unreachable!("Leaf cluster is never empty"));
            while let Some(prev) = items.pop() {
                bottom = bottom.par_merge(prev, cost_matrix);
            }

            ftlog::debug!(
                "Finished aligning leaf cluster at depth {} with {} sequences to {} width in parallel",
                cluster.depth,
                cluster.cardinality,
                bottom.len()
            );

            bottom
        }
    }
}
