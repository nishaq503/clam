//! Ranged Nearest Neighbor (RNN) search with a naive linear scan.

use rayon::prelude::*;

use crate::{cakes::Search, DistanceValue};

/// Ranged Nearest Neighbor (RNN) search with a naive linear scan.
///
/// The field is the radius of the query ball to search within.
pub struct RnnLinear<T: DistanceValue>(pub T);

impl<T: DistanceValue> std::fmt::Display for RnnLinear<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RnnLinear(radius={})", self.0)
    }
}

impl<Id, I, T, A, M> Search<Id, I, T, A, M> for RnnLinear<T>
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    fn search(&self, tree: &crate::tree::Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)> {
        tree.items()
            .iter()
            .enumerate()
            .filter_map(|(idx, (_, item))| {
                let dist = (tree.metric())(query, item);
                if dist <= self.0 {
                    Some((idx, dist))
                } else {
                    None
                }
            })
            .collect()
    }

    fn par_search(&self, tree: &crate::tree::Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)>
    where
        Self: Send + Sync,
        Id: Send + Sync,
        I: Send + Sync,
        T: Send + Sync,
        M: Send + Sync,
        A: Send + Sync,
    {
        tree.items()
            .par_iter()
            .enumerate()
            .filter_map(|(idx, (_, item))| {
                let dist = (tree.metric())(query, item);
                if dist <= self.0 {
                    Some((idx, dist))
                } else {
                    None
                }
            })
            .collect()
    }
}
