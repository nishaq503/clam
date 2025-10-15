//! K-Nearest Neighbor (KNN) search with a naive linear scan.

use crate::{cakes::Search, utils::SizedHeap, DistanceValue, Tree};

/// K-Nearest Neighbor (KNN) search with a naive linear scan.
///
/// The field is the number of nearest neighbors to find (k).
pub struct KnnLinear(pub usize);

impl<Id, I, T, A, M> Search<Id, I, T, A, M> for KnnLinear
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    fn name(&self) -> String {
        format!("KnnLinear(k={})", self.0)
    }

    fn search(&self, tree: &Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)> {
        let mut heap = SizedHeap::new(Some(self.0));
        heap.extend(tree.distances_to_items_in_cluster(query, tree.root()));
        heap.take_items().collect()
    }

    fn par_search(&self, tree: &Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)>
    where
        Self: Send + Sync,
        Id: Send + Sync,
        I: Send + Sync,
        T: Send + Sync,
        A: Send + Sync,
        M: Send + Sync,
    {
        let mut heap = SizedHeap::new(Some(self.0));
        heap.extend(tree.par_distances_to_items_in_cluster(query, tree.root()));
        heap.take_items().collect()
    }
}
