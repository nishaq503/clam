//! A multiple sequence alignment (MSA) of sequences of type `S`.

use rayon::prelude::*;

use crate::{Cluster, DistanceValue, Tree};

mod alignment_ops;
mod columnar;
mod cost_matrix;
mod sequence;

pub use cost_matrix::CostMatrix;
pub use sequence::Sequence;

use alignment_ops::{Direction, Edit, Edits};
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

        let columnar = Self::from_cluster(&tree.root, tree, cost_matrix);
        ftlog::info!("Finished creating Columnar MSA with {} columns", columnar.len());

        let msa = columnar.into_rows(true);
        ftlog::info!("Converted Columnar MSA to {} sequences", msa.len());

        Self(msa)
    }

    /// Recursively creates an MSA from a Cluster.
    fn from_cluster<Id, T, A, M>(cluster: &Cluster<T, A>, tree: &Tree<Id, S, T, A, M>, cost_matrix: &CostMatrix<T>) -> Columnar<S>
    where
        T: DistanceValue,
    {
        if let Some((children, _)) = &cluster.children {
            ftlog::debug!("Aligning parent cluster with {} sequences", cluster.cardinality());

            let mut children = children.iter().map(|child| Self::from_cluster(child, tree, cost_matrix)).collect::<Vec<_>>();

            let mut bottom = children.pop().unwrap_or_else(|| unreachable!("Parent cluster always has children"));
            while let Some(prev) = children.pop() {
                bottom = bottom.merge(prev, cost_matrix);
            }

            bottom.post_pend_row(&tree.items[cluster.center_index].1, cost_matrix)
        } else {
            ftlog::debug!("Aligning leaf cluster with {} sequences", cluster.cardinality());

            let mut items = tree.items[cluster.all_items_indices()]
                .iter()
                .map(|(_, seq)| Columnar::from_row(seq))
                .collect::<Vec<_>>();

            let mut bottom = items.pop().unwrap_or_else(|| unreachable!("Leaf cluster is never empty"));
            while let Some(prev) = items.pop() {
                bottom = bottom.merge(prev, cost_matrix);
            }

            bottom
        }
    }
}

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

        let columnar = Self::par_from_cluster(&tree.root, tree, cost_matrix);
        ftlog::info!("Finished creating Columnar MSA with {} columns in parallel", columnar.len());

        let msa = columnar.par_into_rows(true);
        ftlog::info!("Converted Columnar MSA to {} sequences in parallel", msa.len());

        Self(msa)
    }

    /// Parallel version of [`Self::from_cluster`].
    fn par_from_cluster<Id, T, A, M>(cluster: &Cluster<T, A>, tree: &Tree<Id, S, T, A, M>, cost_matrix: &CostMatrix<T>) -> Columnar<S>
    where
        Id: Send + Sync,
        T: DistanceValue + Send + Sync,
        A: Send + Sync,
        M: Send + Sync,
    {
        if let Some((children, _)) = &cluster.children {
            ftlog::debug!("Aligning parent cluster with {} sequences in parallel", cluster.cardinality());

            let mut children = children
                .par_iter()
                .map(|child| Self::par_from_cluster(child, tree, cost_matrix))
                .collect::<Vec<_>>();

            let mut bottom = children.pop().unwrap_or_else(|| unreachable!("Parent cluster always has children"));
            while let Some(prev) = children.pop() {
                bottom = bottom.par_merge(prev, cost_matrix);
            }

            bottom.post_pend_row(&tree.items[cluster.center_index].1, cost_matrix)
        } else {
            ftlog::debug!("Aligning leaf cluster with {} sequences in parallel", cluster.cardinality());

            let mut items = tree.items[cluster.all_items_indices()]
                .par_iter()
                .map(|(_, seq)| Columnar::from_row(seq))
                .collect::<Vec<_>>();

            let mut bottom = items.pop().unwrap_or_else(|| unreachable!("Leaf cluster is never empty"));
            while let Some(prev) = items.pop() {
                bottom = bottom.par_merge(prev, cost_matrix);
            }

            bottom
        }
    }
}
