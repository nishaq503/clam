//! Ranged Nearest Neighbor (RNN) search with a naive linear scan.

use rayon::prelude::*;

use crate::{
    cakes::{ParSearch, Search},
    Cluster, DistanceValue,
};

/// Ranged Nearest Neighbor (RNN) search with a naive linear scan.
pub struct RnnLinear<T: DistanceValue>(pub T);

impl<Id, I, T: DistanceValue, M: Fn(&I, &I) -> T, A> Search<Id, I, T, M, A> for RnnLinear<T> {
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
}

impl<
        Id: Send + Sync,
        I: Send + Sync,
        T: DistanceValue + Send + Sync,
        M: Fn(&I, &I) -> T + Send + Sync,
        A: Send + Sync,
    > ParSearch<Id, I, T, M, A> for RnnLinear<T>
{
    fn par_search<'a>(&self, root: &'a Cluster<Id, I, T, A>, metric: &M, query: &I) -> Vec<(&'a Id, &'a I, T)> {
        root.all_items()
            .into_par_iter()
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
