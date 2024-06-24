//! CHAODA ensemble on a single tree.

use distances::Number;

use crate::{
    chaoda::algorithms::{
        ClusterCardinality, GraphNeighborhood, ParentCardinality, StationaryProbability, SubgraphCardinality,
        VertexDegree,
    },
    Dataset, Instance, PartitionCriterion,
};

use super::{Algorithm, Graph, OddBall};

/// A CHAODA ensemble on a single tree.
///
/// # Type Parameters
///
/// * `I`: The type of the instances in the dataset.
/// * `U`: The type of the distance values in the dataset.
/// * `D`: The type of the dataset.
/// * `C`: The type of the `OddBall` in the ensemble.
/// * `N`: The number of anomaly ratios in the `OddBall`.
pub struct SingleTreeChaoda<I, U, D, C, const N: usize>
where
    I: Instance,
    U: Number,
    D: Dataset<I, U>,
    C: OddBall<U, N>,
{
    /// The data.
    data: D,
    /// The root `Cluster` of the tree.
    root: C,
    /// The algorithms in the ensemble.
    algorithms: Vec<Box<dyn Algorithm<U>>>,
    /// Phantom data to satisfy the compiler.
    _phantom: std::marker::PhantomData<I>,
    /// The `Graph`s in the ensemble.
    graphs: Vec<Graph<U>>,
}

impl<I, U, D, C, const N: usize> SingleTreeChaoda<I, U, D, C, N>
where
    I: Instance,
    U: Number,
    D: Dataset<I, U>,
    C: OddBall<U, N>,
{
    /// Create a new `SingleTreeChaoda` ensemble.
    pub fn new<P: PartitionCriterion<U>>(
        mut data: D,
        criteria: &P,
        seed: Option<u64>,
        algs: Option<Vec<Box<dyn Algorithm<U>>>>,
    ) -> Self {
        let root = C::new_root(&data, seed).partition(&mut data, criteria, seed);
        let algorithms = algs.unwrap_or_else(|| Self::default_algorithms());
        Self {
            data,
            root,
            algorithms,
            _phantom: std::marker::PhantomData,
            graphs: Vec::new(),
        }
    }

    /// Create the default algorithms for the CHAODA ensemble.
    fn default_algorithms() -> Vec<Box<dyn Algorithm<U>>> {
        let cc = ClusterCardinality;
        let gn = GraphNeighborhood::new(0.1).unwrap_or_else(|_| unreachable!("We chose the neighborhood radius"));
        let pc = ParentCardinality;
        let sc = SubgraphCardinality;
        let sp = StationaryProbability::new(16);
        let vd = VertexDegree;

        vec![
            Box::new(cc),
            Box::new(gn),
            Box::new(pc),
            Box::new(sc),
            Box::new(sp),
            Box::new(vd),
        ]
    }

    /// Get the root `Cluster` of the tree.
    pub const fn root(&self) -> &C {
        &self.root
    }

    /// Get the data.
    pub const fn data(&self) -> &D {
        &self.data
    }

    /// Get the algorithms in the ensemble.
    pub fn algorithms(&self) -> &[Box<dyn Algorithm<U>>] {
        &self.algorithms
    }

    /// Get the `Graph`s in the ensemble.
    pub fn graphs(&self) -> &[Graph<U>] {
        &self.graphs
    }
}
