//! K-Nearest Neighbor (KNN) search with a naive linear scan.

use rayon::prelude::*;

use crate::{
    cakes::{ParSearch, Search},
    utils::SizedHeap,
    Cluster, DistanceValue,
};

/// K-Nearest Neighbor (KNN) search with a naive linear scan.
pub struct KnnLinear(pub usize);

impl std::fmt::Display for KnnLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KnnLinear(k={})", self.0)
    }
}

impl<Id, I, T: DistanceValue, M: Fn(&I, &I) -> T, A> Search<Id, I, T, M, A> for KnnLinear {
    fn search<'a>(&self, root: &'a Cluster<Id, I, T, A>, metric: &M, query: &I) -> Vec<(&'a Id, &'a I, T)> {
        let mut heap = SizedHeap::new(Some(self.0));
        heap.extend(root.all_items().into_iter().map(|item| (item, metric(query, &item.1))));
        heap.items().map(|((id, item), d)| (id, item, d)).collect()
    }

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
                heap.items().map(|((id, item), d)| (id, item, d)).collect()
            })
            .collect()
    }
}

impl<Id, I, T, M, A> ParSearch<Id, I, T, M, A> for KnnLinear
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
    A: Send + Sync,
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

    fn par_batch_search<'a>(
        &self,
        root: &'a Cluster<Id, I, T, A>,
        metric: &M,
        queries: &[I],
    ) -> Vec<Vec<(&'a Id, &'a I, T)>> {
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
                heap.items().map(|((id, item), d)| (id, item, d)).collect()
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
        M: Send + Sync,
    {
        self.par_batch_search(root, metric, queries)
    }
}
