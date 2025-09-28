//! K-Nearest Neighbor (KNN) search with a naive linear scan.

use rayon::prelude::*;

use crate::{utils::SizedHeap, Cluster, DistanceValue};

use super::{ParSearch, Search};

/// K-Nearest Neighbor (KNN) search with a naive linear scan.
pub struct KnnLinear(pub usize);

impl<Id, I, T: DistanceValue, M: Fn(&I, &I) -> T, A> Search<Id, I, T, M, A> for KnnLinear {
    fn search<'a>(&self, root: &'a Cluster<Id, I, T, A>, metric: &M, query: &I) -> Vec<(&'a Id, &'a I, T)> {
        let mut heap = SizedHeap::new(Some(self.0));
        heap.extend(root.all_items().into_iter().map(|item| (item, metric(query, &item.1))));
        heap.items().map(|((id, item), d)| (id, item, d)).collect()
    }
}

impl<
        I: Send + Sync,
        Id: Send + Sync,
        T: DistanceValue + Send + Sync,
        M: Fn(&I, &I) -> T + Send + Sync,
        A: Send + Sync,
    > ParSearch<Id, I, T, M, A> for KnnLinear
{
    fn par_search<'a>(&self, root: &'a Cluster<Id, I, T, A>, metric: &M, query: &I) -> Vec<(&'a Id, &'a I, T)> {
        let mut heap = SizedHeap::new(Some(self.0));
        heap.extend(
            root.all_items()
                .into_par_iter()
                .map(|item| (item, metric(query, &item.1)))
                .collect::<Vec<_>>(),
        );
        heap.items().map(|((id, item), d)| (id, item, d)).collect()
    }
}
