//! Ranged Nearest Neighbor (RNN) search with a naive linear scan.

use rayon::prelude::*;

use crate::{
    cakes::{BatchedSearch, Search},
    Cluster, DistanceValue,
};

/// Ranged Nearest Neighbor (RNN) search with a naive linear scan.
///
/// The field is the radius of the query ball to search within.
pub struct RnnLinear<T: DistanceValue>(pub T);

impl<T: DistanceValue> std::fmt::Display for RnnLinear<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RnnLinear(radius={})", self.0)
    }
}

impl<Id, I, T: DistanceValue, A, M: Fn(&I, &I) -> T> Search<Id, I, T, A, M> for RnnLinear<T> {
    fn search<'a>(&self, root: &'a Cluster<Id, I, T, A>, metric: &M, query: &I) -> Vec<(&'a Id, &'a I, T)> {
        root.all_items()
            .into_iter()
            .filter_map(|(id, item)| {
                let dist = metric(query, item);
                if dist <= self.0 {
                    Some((id, item, dist))
                } else {
                    None
                }
            })
            .collect()
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
        root.all_items()
            .par_iter()
            .filter_map(|(id, item)| {
                let dist = metric(query, item);
                if dist <= self.0 {
                    Some((id, item, dist))
                } else {
                    None
                }
            })
            .collect()
    }
}

impl<Id, I, T: DistanceValue, A, M: Fn(&I, &I) -> T> BatchedSearch<Id, I, T, A, M> for RnnLinear<T> {
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
                all_items
                    .iter()
                    .filter_map(|(id, item)| {
                        let dist = metric(query, item);
                        if dist <= self.0 {
                            Some((id, item, dist))
                        } else {
                            None
                        }
                    })
                    .collect()
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
                all_items
                    .par_iter()
                    .filter_map(|(id, item)| {
                        let dist = metric(query, item);
                        if dist <= self.0 {
                            Some((id, item, dist))
                        } else {
                            None
                        }
                    })
                    .collect()
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
