//! K-Nearest Neighbor (KNN) search with a naive linear scan.

use rayon::prelude::*;

use crate::{
    cakes::{BatchedSearch, Search},
    utils::SizedHeap,
    Cluster, DistanceValue,
};

/// K-Nearest Neighbor (KNN) search with a naive linear scan.
///
/// The field is the number of nearest neighbors to find (k).
pub struct KnnLinear(pub usize);

impl std::fmt::Display for KnnLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KnnLinear(k={})", self.0)
    }
}

impl<Id, I, T: DistanceValue, A, M: Fn(&I, &I) -> T> Search<Id, I, T, A, M> for KnnLinear {
    fn search<'a>(&self, root: &'a Cluster<Id, I, T, A>, metric: &M, query: &I) -> Vec<(&'a Id, &'a I, T)> {
        let mut heap = SizedHeap::new(Some(self.0));
        heap.extend(root.all_items().into_iter().map(|item| (item, metric(query, &item.1))));
        heap.take_items().map(|((id, item), d)| (id, item, d)).collect()
    }

    fn par_search<'a>(&self, root: &'a Cluster<Id, I, T, A>, metric: &M, query: &I) -> Vec<(&'a Id, &'a I, T)>
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
            root.all_items()
                .into_par_iter()
                .map(|item| (item, metric(query, &item.1)))
                .collect::<Vec<_>>(),
        );
        heap.take_items().map(|((id, item), d)| (id, item, d)).collect()
    }
}

impl<Id, I, T: DistanceValue, A, M: Fn(&I, &I) -> T> BatchedSearch<Id, I, T, A, M> for KnnLinear {
    fn batch_search<'a>(
        &self,
        root: &'a Cluster<Id, I, T, A>,
        metric: &M,
        queries: &[I],
    ) -> Vec<Vec<(&'a Id, &'a I, T)>> {
        let all_items = root.all_items();
        queries
            .iter()
            .map(|query| {
                let mut heap = SizedHeap::new(Some(self.0));
                heap.extend(all_items.iter().map(|item| (item, metric(query, &item.1))));
                heap.take_items().map(|((id, item), d)| (id, item, d)).collect()
            })
            .collect()
    }

    fn par_batch_search<'a>(
        &self,
        root: &'a Cluster<Id, I, T, A>,
        metric: &M,
        queries: &[I],
    ) -> Vec<Vec<(&'a Id, &'a I, T)>>
    where
        Self: Send + Sync,
        Id: Send + Sync,
        I: Send + Sync,
        T: Send + Sync,
        A: Send + Sync,
        M: Send + Sync,
    {
        let all_items = root.all_items();
        queries
            .par_iter()
            .map(|query| {
                let mut heap = SizedHeap::new(Some(self.0));
                heap.extend(
                    all_items
                        .par_iter()
                        .map(|item| (item, metric(query, &item.1)))
                        .collect::<Vec<_>>(),
                );
                heap.take_items().map(|((id, item), d)| (id, item, d)).collect()
            })
            .collect()
    }

    fn par_batch_par_search<'a>(
        &self,
        root: &'a Cluster<Id, I, T, A>,
        metric: &M,
        queries: &[I],
    ) -> Vec<Vec<(&'a Id, &'a I, T)>>
    where
        Self: Send + Sync,
        Id: Send + Sync,
        I: Send + Sync,
        T: Send + Sync,
        A: Send + Sync,
        M: Send + Sync,
    {
        self.par_batch_search(root, metric, queries)
    }
}
