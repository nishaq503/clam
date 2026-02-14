//! Compressive search algorithms.

use crate::{DistanceValue, Tree};

use super::{Codec, MaybeCompressed};

// mod approximate;
mod exact;
// mod selection;

pub use exact::RnnChess;
// pub use exact::{KnnBfs, KnnDfs, KnnRrnn, RnnChess};
// pub(crate) use exact::{leaf_into_hits, pop_till_leaf};

/// `PanCakes` algorithms.
pub enum PanCakes<T: DistanceValue> {
    // /// K-Nearest Neighbors Breadth-First Sieve.
    // KnnBfs(KnnBfs),
    // /// K-Nearest Neighbors Depth-First Sieve.
    // KnnDfs(KnnDfs),
    // /// K-Nearest Neighbors Repeated RNN.
    // KnnRrnn(KnnRrnn),
    /// Ranged Nearest Neighbors Chess Search.
    RnnChess(RnnChess<T>),
    // /// Approximate K-Nearest Neighbors Depth-First Sieve.
    // ApproxKnnDfs(approximate::KnnDfs),
}

impl<T: DistanceValue> PanCakes<T> {
    /// Returns the name of the algorithm.
    pub fn name(&self) -> String {
        match self {
            // Self::KnnBfs(KnnBfs(k)) => format!("KnnBfs(k={k})"),
            // Self::KnnDfs(KnnDfs(k)) => format!("KnnDfs(k={k})"),
            // Self::KnnRrnn(KnnRrnn(k)) => format!("KnnRrnn(k={k})"),
            Self::RnnChess(RnnChess(r)) => format!("RnnChess(r={r})"),
            // Self::ApproxKnnDfs(alg) => format!("{alg}"),
        }
    }
}

/// A Nearest Neighbor Search algorithm in compressed space.
pub trait CompressiveSearch<Id, I, T, A, M>
where
    I: Codec,
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    /// Returns a name for the search algorithm.
    ///
    /// This is intended for diagnostic use. Ideally, it should include information about the parameters of the algorithm.
    fn name(&self) -> String;

    /// Searches for nearest neighbors of `query` in the given `tree` and returns a vector of `(index, distance)` pairs into the `items` of the `tree`.
    fn search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Vec<(usize, T)>;
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
    fn par_search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Vec<(usize, T)>;
}

impl<Id, I, T, A, M> CompressiveSearch<Id, I, T, A, M> for PanCakes<T>
where
    I: Codec,
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    fn name(&self) -> String {
        match self {
            // Self::KnnBfs(alg) => <KnnBfs as CompressiveSearch<Id, I, T, A, M>>::name(alg),
            // Self::KnnDfs(alg) => <KnnDfs as CompressiveSearch<Id, I, T, A, M>>::name(alg),
            // Self::KnnRrnn(alg) => <KnnRrnn as CompressiveSearch<Id, I, T, A, M>>::name(alg),
            Self::RnnChess(alg) => <RnnChess<T> as CompressiveSearch<Id, I, T, A, M>>::name(alg),
            // Self::ApproxKnnDfs(alg) => <approximate::KnnDfs as CompressiveSearch<Id, I, T, A, M>>::name(alg),
        }
    }

    fn search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Vec<(usize, T)> {
        match self {
            // Self::KnnBfs(alg) => alg.search(tree, query),
            // Self::KnnDfs(alg) => alg.search(tree, query),
            // Self::KnnRrnn(alg) => alg.search(tree, query),
            Self::RnnChess(alg) => alg.search(tree, query),
            // Self::ApproxKnnDfs(alg) => alg.search(tree, query),
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
    fn par_search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Vec<(usize, T)> {
        match self {
            // Self::KnnBfs(alg) => alg.par_search(tree, query),
            // Self::KnnDfs(alg) => alg.par_search(tree, query),
            // Self::KnnRrnn(alg) => alg.par_search(tree, query),
            Self::RnnChess(alg) => alg.par_search(tree, query),
            // Self::ApproxKnnDfs(alg) => alg.par_search(tree, query),
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
    fn name(&self) -> String {
        (**self).name()
    }

    fn search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Vec<(usize, T)> {
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
    fn name(&self) -> String {
        (**self).name()
    }

    fn search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Vec<(usize, T)> {
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
    fn par_search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Vec<(usize, T)> {
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
    fn par_search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Vec<(usize, T)> {
        (**self).par_search(tree, query)
    }
}
