//! K-Nearest Neighbor (KNN) search with a naive linear scan.

use rayon::prelude::*;

use crate::{tree::cakes::Search, utils::SizedHeap, DistanceValue};

/// K-Nearest Neighbor (KNN) search with a naive linear scan.
///
/// The field is the number of nearest neighbors to find (k).
pub struct KnnLinear(pub usize);

impl<Id, I, T, A, M> Search<Id, I, T, A, M> for KnnLinear
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    fn search(&self, tree: &crate::tree::Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)> {
        let mut heap = SizedHeap::new(Some(self.0));
        heap.extend(
            tree.items()
                .iter()
                .enumerate()
                .map(|(i, (_, item))| (i, tree.metric()(query, item))),
        );
        heap.take_items().collect()
    }

    fn par_search(&self, tree: &crate::tree::Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)>
    where
        Self: Send + Sync,
        Id: Send + Sync,
        I: Send + Sync,
        T: Send + Sync,
        A: Send + Sync,
        M: Send + Sync,
    {
        let mut heap = SizedHeap::new(Some(self.0));
        heap.extend(
            tree.items()
                .into_par_iter()
                .enumerate()
                .map(|(i, (_, item))| (i, tree.metric()(query, item)))
                .collect::<Vec<_>>(),
        );
        heap.take_items().collect()
    }
}
