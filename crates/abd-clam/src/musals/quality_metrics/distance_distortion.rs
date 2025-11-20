//! Distance Distortion MSA Quality Metric

use rayon::prelude::*;

use crate::{
    DistanceValue, Tree,
    musals::{CostMatrix, Sequence},
};

use super::{
    MsaQuality, mu_sigma_min_max,
    pairwise_scores::{apply_pairwise, par_apply_pairwise},
};

/// The scores of pairwise alignments in the MSA.
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct DistanceDistortion {
    /// Mean
    mean: f64,
    /// Standard Deviation
    std_dev: f64,
    /// Minimum
    min: f64,
    /// Maximum
    max: f64,
}

impl MsaQuality for DistanceDistortion {
    fn name(&self) -> String {
        "DistanceDistortion".to_string()
    }

    fn short_name<'a>(&self) -> &'a str {
        "dd"
    }

    fn description(&self) -> String {
        "The mean distortion of alignment distances between pairs of sequences in the MSA.".to_string()
    }

    fn mean(&self) -> f64 {
        self.mean
    }

    fn std_dev(&self) -> f64 {
        self.std_dev
    }

    fn min(&self) -> f64 {
        self.min
    }

    fn max(&self) -> f64 {
        self.max
    }

    fn compute<Id, S, T, A, M>(msa_tree: &Tree<Id, S, T, A, M>, _: &CostMatrix<T>) -> Self
    where
        S: Sequence,
        T: DistanceValue,
        M: Fn(&S, &S) -> T,
        Self: Sized,
    {
        let scorer = |(_, s1): &(Id, S), (_, s2): &(Id, S)| dd_inner(s1, s2, &msa_tree.metric);
        let indices = (0..msa_tree.cardinality()).collect::<Vec<_>>();
        let pairwise_scores = apply_pairwise(&msa_tree.items, &indices, scorer).collect::<Vec<_>>();
        let (mean, std_dev, min, max) = mu_sigma_min_max(&pairwise_scores);
        Self {
            mean,
            std_dev,
            min,
            max,
        }
    }

    fn par_compute<Id, S, T, A, M>(msa_tree: &Tree<Id, S, T, A, M>, _: &CostMatrix<T>) -> Self
    where
        Id: Send + Sync,
        S: Sequence + Send + Sync,
        T: DistanceValue + Send + Sync,
        A: Send + Sync,
        M: Fn(&S, &S) -> T + Send + Sync,
        Self: Sized + Send + Sync,
    {
        let scorer = |(_, s1): &(Id, S), (_, s2): &(Id, S)| dd_inner(s1, s2, &msa_tree.metric);
        let indices = (0..msa_tree.cardinality()).collect::<Vec<_>>();
        let pairwise_scores = par_apply_pairwise(&msa_tree.items, &indices, scorer).collect::<Vec<_>>();
        let (mean, std_dev, min, max) = mu_sigma_min_max(&pairwise_scores);
        Self {
            mean,
            std_dev,
            min,
            max,
        }
    }
}

/// Measures the distortion of the Levenshtein edit distance between the unaligned sequences and the Hamming distance between the aligned sequences.
fn dd_inner<S: Sequence, T: DistanceValue, M: Fn(&S, &S) -> T>(s1: &S, s2: &S, metric: &M) -> f64 {
    let (s1, s2) = (s1.as_ref(), s2.as_ref());
    let ham = s1.iter().zip(s2.iter()).filter(|(a, b)| a != b).count();

    let s1 = S::from_vec(s1.iter().filter(|&&c| c != S::GAP).copied().collect());
    let s2 = S::from_vec(s2.iter().filter(|&&c| c != S::GAP).copied().collect());
    let m = metric(&s1, &s2);

    if m == T::zero() {
        1.0
    } else {
        ham as f64
            / m.to_f64()
                .unwrap_or_else(|| unreachable!("DistanceValue conversion to f64 failed"))
    }
}
