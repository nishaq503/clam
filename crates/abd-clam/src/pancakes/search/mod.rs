//! Compressive search algorithms.

use crate::{
    DistanceValue, Tree,
    cakes::{KnnBfs, KnnDfs, KnnRrnn, RnnChess, approximate::KnnDfs as ApproxKnnDfs},
};

use super::{Codec, MaybeCompressed};

mod approximate;
mod exact;
// mod selection;

pub(crate) use exact::{leaf_into_hits, par_leaf_into_hits, par_pop_till_leaf, pop_till_leaf};

/// `PanCakes` algorithms.
pub enum PanCakes<T: DistanceValue> {
    /// K-Nearest Neighbors Breadth-First Sieve.
    KnnBfs(KnnBfs),
    /// K-Nearest Neighbors Depth-First Sieve.
    KnnDfs(KnnDfs),
    /// K-Nearest Neighbors Repeated RNN.
    KnnRrnn(KnnRrnn),
    /// Ranged Nearest Neighbors Chess Search.
    RnnChess(RnnChess<T>),
    /// Approximate K-Nearest Neighbors Depth-First Sieve.
    ApproxKnnDfs(ApproxKnnDfs),
}

impl<T: DistanceValue> PanCakes<T> {
    /// Returns the name of the algorithm.
    pub fn name(&self) -> String {
        match self {
            Self::KnnBfs(KnnBfs(k)) => format!("KnnBfs(k={k})"),
            Self::KnnDfs(KnnDfs(k)) => format!("KnnDfs(k={k})"),
            Self::KnnRrnn(KnnRrnn(k)) => format!("KnnRrnn(k={k})"),
            Self::RnnChess(RnnChess(r)) => format!("RnnChess(r={r})"),
            Self::ApproxKnnDfs(alg) => format!("{alg}"),
        }
    }
}

/// A Nearest Neighbor Search algorithm in compressed space, decompressing items as needed.
pub trait CompressiveSearch<Id, I, T, A, M>
where
    I: Codec,
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    /// Same as [`Search::search`](crate::cakes::Search::search) but operates on a compressed tree and will decompress items as needed.
    ///
    /// # Errors
    ///
    /// - If the root center has been compressed.
    fn search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Result<Vec<(usize, T)>, String>;
}

/// Parallel version of [`CompressiveSearch`].
pub trait ParCompressiveSearch<Id, I, T, A, M>: CompressiveSearch<Id, I, T, A, M> + Send + Sync
where
    Id: Send + Sync,
    I: Codec + Send + Sync,
    I::Compressed: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
{
    /// Parallel version of [`CompressiveSearch::search`].
    ///
    /// # Errors
    ///
    /// See [`CompressiveSearch::search`] for error conditions.
    fn par_search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Result<Vec<(usize, T)>, String>;
}

impl<Id, I, T, A, M> CompressiveSearch<Id, I, T, A, M> for PanCakes<T>
where
    I: Codec,
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    fn search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Result<Vec<(usize, T)>, String> {
        match self {
            Self::KnnBfs(alg) => alg.search(tree, query),
            Self::KnnDfs(alg) => alg.search(tree, query),
            Self::KnnRrnn(alg) => alg.search(tree, query),
            Self::RnnChess(alg) => alg.search(tree, query),
            Self::ApproxKnnDfs(alg) => alg.search(tree, query),
        }
    }
}

impl<Id, I, T, A, M> ParCompressiveSearch<Id, I, T, A, M> for PanCakes<T>
where
    Id: Send + Sync,
    I: Codec + Send + Sync,
    I::Compressed: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
{
    fn par_search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Result<Vec<(usize, T)>, String> {
        match self {
            Self::KnnBfs(alg) => alg.par_search(tree, query),
            Self::KnnDfs(alg) => alg.par_search(tree, query),
            Self::KnnRrnn(alg) => alg.par_search(tree, query),
            Self::RnnChess(alg) => alg.par_search(tree, query),
            Self::ApproxKnnDfs(alg) => alg.par_search(tree, query),
        }
    }
}

// Blanket implementations of `Search` for references and boxes.
impl<Id, I, T, A, M, Alg> CompressiveSearch<Id, I, T, A, M> for &Alg
where
    I: Codec,
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
    Alg: CompressiveSearch<Id, I, T, A, M>,
{
    fn search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Result<Vec<(usize, T)>, String> {
        (**self).search(tree, query)
    }
}

impl<Id, I, T, A, M, Alg> CompressiveSearch<Id, I, T, A, M> for Box<Alg>
where
    I: Codec,
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
    Alg: CompressiveSearch<Id, I, T, A, M>,
{
    fn search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Result<Vec<(usize, T)>, String> {
        (**self).search(tree, query)
    }
}

// Blanket implementations of `ParSearch` for references and boxes.
impl<Id, I, T, A, M, Alg> ParCompressiveSearch<Id, I, T, A, M> for &Alg
where
    Id: Send + Sync,
    I: Codec + Send + Sync,
    I::Compressed: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
    Alg: ParCompressiveSearch<Id, I, T, A, M>,
{
    fn par_search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Result<Vec<(usize, T)>, String> {
        (**self).par_search(tree, query)
    }
}

impl<Id, I, T, A, M, Alg> ParCompressiveSearch<Id, I, T, A, M> for Box<Alg>
where
    Id: Send + Sync,
    I: Codec + Send + Sync,
    I::Compressed: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
    Alg: ParCompressiveSearch<Id, I, T, A, M>,
{
    fn par_search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Result<Vec<(usize, T)>, String> {
        (**self).par_search(tree, query)
    }
}
