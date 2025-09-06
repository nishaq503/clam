//! A multiple sequence alignment (MSA) of sequences of type `S`.

use rayon::prelude::*;

use crate::{Cluster, DistanceValue, Tree};

use super::{Columnar, CostMatrix, Sequence};

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
        let columnar = Self::from_cluster(&tree.root, tree, cost_matrix);
        Self(columnar.into_rows(true))
    }

    /// Recursively creates an MSA from a Cluster.
    fn from_cluster<Id, T, A, M>(
        cluster: &Cluster<T, A>,
        tree: &Tree<Id, S, T, A, M>,
        cost_matrix: &CostMatrix<T>,
    ) -> Columnar<S>
    where
        T: DistanceValue,
    {
        if let Some((children, _)) = &cluster.children {
            let mut children = children
                .iter()
                .map(|child| Self::from_cluster(child, tree, cost_matrix))
                .collect::<Vec<_>>();

            let mut bottom = children
                .pop()
                .unwrap_or_else(|| unreachable!("Parent cluster always has children"));
            while let Some(prev) = children.pop() {
                bottom = bottom.merge(prev, cost_matrix);
            }

            bottom.post_pend_row(&tree.items[cluster.center_index].1, cost_matrix)
        } else {
            let mut items = tree.items[cluster.all_items_indices()]
                .iter()
                .map(|(_, seq)| Columnar::from_row(seq))
                .collect::<Vec<_>>();

            let mut bottom = items
                .pop()
                .unwrap_or_else(|| unreachable!("Leaf cluster is never empty"));
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
        let columnar = Self::par_from_cluster(&tree.root, tree, cost_matrix);
        Self(columnar.par_into_rows(true))
    }

    /// Parallel version of [`Self::from_cluster`].
    fn par_from_cluster<Id, T, A, M>(
        cluster: &Cluster<T, A>,
        tree: &Tree<Id, S, T, A, M>,
        cost_matrix: &CostMatrix<T>,
    ) -> Columnar<S>
    where
        Id: Send + Sync,
        T: DistanceValue + Send + Sync,
        A: Send + Sync,
        M: Send + Sync,
    {
        if let Some((children, _)) = &cluster.children {
            let mut children = children
                .par_iter()
                .map(|child| Self::par_from_cluster(child, tree, cost_matrix))
                .collect::<Vec<_>>();

            let mut bottom = children
                .pop()
                .unwrap_or_else(|| unreachable!("Parent cluster always has children"));
            while let Some(prev) = children.pop() {
                bottom = bottom.par_merge(prev, cost_matrix);
            }

            bottom.post_pend_row(&tree.items[cluster.center_index].1, cost_matrix)
        } else {
            let mut items = tree.items[cluster.all_items_indices()]
                .par_iter()
                .map(|(_, seq)| Columnar::from_row(seq))
                .collect::<Vec<_>>();

            let mut bottom = items
                .pop()
                .unwrap_or_else(|| unreachable!("Leaf cluster is never empty"));
            while let Some(prev) = items.pop() {
                bottom = bottom.par_merge(prev, cost_matrix);
            }

            bottom
        }
    }
}
