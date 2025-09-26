//! K-Nearest Neighbor (KNN) search with a naive linear scan.

use rayon::prelude::*;

use crate::{utils::SizedHeap, Ball, DistanceValue};

use super::{ParSearch, Search};

/// K-Nearest Neighbor (KNN) search with a naive linear scan.
pub struct KnnLinear(pub usize);

impl<I, T: DistanceValue, M: Fn(&I, &I) -> T> Search<I, T, M> for KnnLinear {
    fn search<'a>(&self, root: &'a Ball<I, T>, metric: &M, query: &I) -> Vec<(&'a I, T)> {
        let mut heap = SizedHeap::new(Some(self.0));
        heap.extend(root.all_items().into_iter().map(|item| (item, metric(query, item))));
        heap.items().collect()
    }
}

impl<I: Send + Sync, T: DistanceValue + Send + Sync, M: Fn(&I, &I) -> T + Send + Sync> ParSearch<I, T, M> for KnnLinear {
    fn par_search<'a>(&self, root: &'a Ball<I, T>, metric: &M, query: &I) -> Vec<(&'a I, T)> {
        let mut heap = SizedHeap::new(Some(self.0));
        heap.extend(
            root.all_items()
                .into_par_iter()
                .map(|item| (item, metric(query, item)))
                .collect::<Vec<_>>(),
        );
        heap.items().collect()
    }
}
