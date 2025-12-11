//! Scores each pairwise alignment in the MSA, applying a penalty for gaps and mismatches.

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
pub struct WeightedPairwiseScores {
    /// Mean
    mean: f64,
    /// Standard Deviation
    std_dev: f64,
    /// Minimum
    min: f64,
    /// Maximum
    max: f64,
}

impl MsaQuality for WeightedPairwiseScores {
    fn name(&self) -> String {
        "WeightedPairwiseScores".to_string()
    }

    fn short_name<'a>(&self) -> &'a str {
        "wps"
    }

    fn description(&self) -> String {
        "The mean score of all pairwise alignments, weighted gap extensions, in the MSA.".to_string()
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

    fn compute<Id, S, T, A, M>(msa_tree: &Tree<Id, S, T, A, M>, cost_matrix: &CostMatrix<T>) -> Self
    where
        S: Sequence,
        T: DistanceValue,
        M: Fn(&S, &S) -> T,
        Self: Sized,
    {
        let scorer = |(_, s1): &(Id, S), (_, s2): &(Id, S)| wsp_inner(s1, s2, cost_matrix);
        let indices = (0..msa_tree.cardinality()).collect::<Vec<_>>();
        let pairwise_scores = apply_pairwise(&msa_tree.items, &indices, scorer).collect::<Vec<_>>();
        let (mean, std_dev, min, max) = mu_sigma_min_max(&pairwise_scores);
        Self { mean, std_dev, min, max }
    }

    fn par_compute<Id, S, T, A, M>(msa_tree: &Tree<Id, S, T, A, M>, cost_matrix: &CostMatrix<T>) -> Self
    where
        Id: Send + Sync,
        S: Sequence + Send + Sync,
        T: DistanceValue + Send + Sync,
        A: Send + Sync,
        M: Fn(&S, &S) -> T + Send + Sync,
        Self: Sized + Send + Sync,
    {
        let scorer = |(_, s1): &(Id, S), (_, s2): &(Id, S)| wsp_inner(s1, s2, cost_matrix);
        let indices = (0..msa_tree.cardinality()).collect::<Vec<_>>();
        let pairwise_scores = par_apply_pairwise(&msa_tree.items, &indices, scorer).collect::<Vec<_>>();
        let (mean, std_dev, min, max) = mu_sigma_min_max(&pairwise_scores);
        Self { mean, std_dev, min, max }
    }
}

/// Scores a single pairwise alignment in the MSA, applying a penalty for opening a gap, extending a gap, and mismatches.
fn wsp_inner<S: Sequence, T: DistanceValue>(s1: &S, s2: &S, cost_matrix: &CostMatrix<T>) -> f64 {
    // let (s1, s2) = remove_gap_only_cols(s1, s2, gap_char);
    let (s1, s2) = (s1.as_ref(), s2.as_ref());

    let start = if s1[0] == s2[0] {
        T::zero()
    } else if s1[0] == S::GAP || s2[0] == S::GAP {
        cost_matrix.gap_open_cost()
    } else {
        // mismatch
        cost_matrix.sub_cost(s1[0], s2[0])
    };

    let score = s1
        .iter()
        .zip(s1.iter().skip(1))
        .zip(s2.iter().zip(s1.iter().skip(1)))
        .fold(start, |score, ((&a1, &a2), (&b1, &b2))| {
            if (a2 == S::GAP && a1 != S::GAP) || (b2 == S::GAP && b1 != S::GAP) {
                score + cost_matrix.gap_open_cost()
            } else if a2 == S::GAP || b2 == S::GAP {
                score + cost_matrix.gap_ext_cost()
            } else if a2 != b2 {
                score + cost_matrix.sub_cost(a2, b2)
            } else {
                score
            }
        });

    score.to_f64().unwrap_or_else(|| unreachable!("DistanceValue to_f64 conversion failed"))
}
