//! Ranged Nearest Neighbor (RNN) search with a naive linear scan.

use rayon::prelude::*;

use crate::{Ball, DistanceValue};

use super::{ParSearch, Search};

/// Ranged Nearest Neighbor (RNN) search with a naive linear scan.
pub struct RnnLinear<T: DistanceValue>(pub T);

impl<I, T: DistanceValue, M: Fn(&I, &I) -> T> Search<I, T, M> for RnnLinear<T> {
    fn search<'a>(&self, root: &'a Ball<I, T>, metric: &M, query: &I) -> Vec<(&'a I, T)> {
        root.all_items()
            .into_iter()
            .filter_map(|item| {
                let dist = metric(query, item);
                if dist <= self.0 {
                    Some((item, dist))
                } else {
                    None
                }
            })
            .collect()
    }
}

impl<I: Send + Sync, T: DistanceValue + Send + Sync, M: Fn(&I, &I) -> T + Send + Sync> ParSearch<I, T, M>
    for RnnLinear<T>
{
    fn par_search<'a>(&self, root: &'a Ball<I, T>, metric: &M, query: &I) -> Vec<(&'a I, T)> {
        root.all_items()
            .into_par_iter()
            .filter_map(|item| {
                let dist = metric(query, item);
                if dist <= self.0 {
                    Some((item, dist))
                } else {
                    None
                }
            })
            .collect()
    }
}
