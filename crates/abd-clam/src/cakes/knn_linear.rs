//! K-Nearest Neighbor (KNN) search with a naive linear scan.

use rayon::prelude::*;

use crate::{utils::SizedHeap, Ball, DistanceValue};

use super::{ParSearch, Search};

/// K-Nearest Neighbor (KNN) search with a naive linear scan.
pub struct KnnLinear(pub usize);

impl<Id, I, T: DistanceValue, M: Fn(&I, &I) -> T> Search<Id, I, T, M> for KnnLinear {
    fn search<'a>(&self, root: &'a Ball<Id, I, T>, metric: &M, query: &I) -> Vec<(&'a (Id, I), T)> {
        let mut heap = SizedHeap::new(Some(self.0));
        heap.extend(root.all_items().into_iter().map(|item| (item, metric(query, &item.1))));
        heap.items().collect()
    }
}

impl<I: Send + Sync, Id: Send + Sync, T: DistanceValue + Send + Sync, M: Fn(&I, &I) -> T + Send + Sync>
    ParSearch<Id, I, T, M> for KnnLinear
{
    fn par_search<'a>(&self, root: &'a Ball<Id, I, T>, metric: &M, query: &I) -> Vec<(&'a (Id, I), T)> {
        let mut heap = SizedHeap::new(Some(self.0));
        heap.extend(
            root.all_items()
                .into_par_iter()
                .map(|item| (item, metric(query, &item.1)))
                .collect::<Vec<_>>(),
        );
        heap.items().collect()
    }
}
