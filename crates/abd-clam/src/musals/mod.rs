//! Multiple Sequence Alignment At Scale (`MuSAlS`) with CLAM.

use rayon::prelude::*;

use crate::{DistanceValue, Tree};

mod alignment;
mod quality_metrics;

pub use alignment::{CostMatrix, Direction, Sequence};
pub use quality_metrics::{DistanceDistortion, GapFraction, MismatchFraction, PairwiseScores, QualityMetric, QualityMetricResult, WeightedPairwiseScores};

use alignment::Msa;
use quality_metrics::MsaQuality;

/// Extension of [`Tree`], gated behind the `musals` feature, providing methods for computing multiple sequence alignments and computing MSA quality metrics.
///
/// Note the use of `S` for `Sequence` instead of `I`. See the [`Sequence`] trait documentation for more information.
impl<Id, S, T, A, M> Tree<Id, S, T, A, M>
where
    S: Sequence,
    T: DistanceValue,
{
    /// Returns a new tree containing the multiple sequence alignment of the sequences in the original tree.
    pub fn into_msa(mut self, cost_matrix: &CostMatrix<T>) -> Self {
        ftlog::info!("Computing MSA for tree with {} sequences.", self.cardinality());
        let msa = Msa::from_tree(&self, cost_matrix);
        self.items = self.items.into_iter().zip(msa).map(|((id, _), aligned_seq)| (id, aligned_seq)).collect();
        self
    }

    /// Same as [`Self::into_msa`], but uses an iterative approach to reduce stack usage.
    pub fn into_msa_iterative(mut self, cost_matrix: &CostMatrix<T>) -> Self {
        ftlog::info!("Computing MSA for tree with {} sequences.", self.cardinality());
        let msa = Msa::from_tree_iterative(&self, cost_matrix);
        self.items = self.items.into_iter().zip(msa).map(|((id, _), aligned_seq)| (id, aligned_seq)).collect();
        self
    }

    /// Computes all quality metrics for the MSA represented by this tree.
    pub fn compute_quality_metric(&self, quality_metric: &QualityMetric, cost_matrix: &CostMatrix<T>) -> QualityMetricResult
    where
        M: Fn(&S, &S) -> T,
    {
        match quality_metric {
            QualityMetric::GapFraction => QualityMetricResult::GapFraction(GapFraction::compute(self, cost_matrix)),
            QualityMetric::MismatchFraction => QualityMetricResult::MismatchFraction(MismatchFraction::compute(self, cost_matrix)),
            QualityMetric::PairwiseScores => QualityMetricResult::PairwiseScores(PairwiseScores::compute(self, cost_matrix)),
            QualityMetric::WeightedPairwiseScores => QualityMetricResult::WeightedPairwiseScores(WeightedPairwiseScores::compute(self, cost_matrix)),
            QualityMetric::DistanceDistortion => QualityMetricResult::DistanceDistortion(DistanceDistortion::compute(self, cost_matrix)),
        }
    }
}

/// Parallel versions of the MSA methods for the `musals` feature.
impl<Id, S, T, A, M> Tree<Id, S, T, A, M>
where
    Id: Send + Sync,
    S: Sequence + Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Send + Sync,
{
    /// Parallel version of [`Self::into_msa`].
    pub fn par_into_msa(mut self, cost_matrix: &CostMatrix<T>) -> Self {
        ftlog::info!("Computing MSA for tree with {} sequences in parallel.", self.cardinality());
        let msa = Msa::par_from_tree(&self, cost_matrix);
        self.items = self.items.into_par_iter().zip(msa).map(|((id, _), aligned_seq)| (id, aligned_seq)).collect();
        self
    }

    /// Parallel version of [`Self::into_msa_iterative`].
    pub fn par_into_msa_iterative(mut self, cost_matrix: &CostMatrix<T>) -> Self {
        ftlog::info!("Computing MSA for tree with {} sequences in parallel.", self.cardinality());
        let msa = Msa::par_from_tree_iterative(&self, cost_matrix);
        self.items = self.items.into_par_iter().zip(msa).map(|((id, _), aligned_seq)| (id, aligned_seq)).collect();
        self
    }

    /// Parallel version of [`Self::compute_quality_metric`].
    pub fn par_compute_quality_metric(&self, quality_metric: &QualityMetric, cost_matrix: &CostMatrix<T>) -> QualityMetricResult
    where
        M: Fn(&S, &S) -> T,
    {
        match quality_metric {
            QualityMetric::GapFraction => QualityMetricResult::GapFraction(GapFraction::par_compute(self, cost_matrix)),
            QualityMetric::MismatchFraction => QualityMetricResult::MismatchFraction(MismatchFraction::par_compute(self, cost_matrix)),
            QualityMetric::PairwiseScores => QualityMetricResult::PairwiseScores(PairwiseScores::par_compute(self, cost_matrix)),
            QualityMetric::WeightedPairwiseScores => QualityMetricResult::WeightedPairwiseScores(WeightedPairwiseScores::par_compute(self, cost_matrix)),
            QualityMetric::DistanceDistortion => QualityMetricResult::DistanceDistortion(DistanceDistortion::par_compute(self, cost_matrix)),
        }
    }
}

#[cfg(test)]
mod tests {
    use distances::strings::levenshtein;
    use rand::prelude::*;
    use test_case::test_case;

    use crate::Tree;

    use super::{CostMatrix, Sequence};

    fn check_sequences_equal<Id, S, T, A, M>(original: &Tree<Id, S, T, A, M>, aligned: &Tree<Id, S, T, A, M>, mode: &str)
    where
        Id: Eq + core::fmt::Debug,
        S: Sequence + Eq + core::fmt::Debug,
    {
        assert_eq!(
            original.cardinality(),
            aligned.cardinality(),
            "Number of sequences should match in {} mode.",
            mode
        );

        let max_len = original.items.iter().map(|(_, seq)| seq.as_ref().len()).max().unwrap_or(0);
        let aligned_max_len = aligned.items.iter().map(|(_, seq)| seq.as_ref().len()).max().unwrap_or(0);
        assert!(
            aligned_max_len >= max_len,
            "Aligned sequences should be at least as long as the longest original sequence in {} mode.",
            mode
        );
        assert!(
            aligned_max_len <= max_len * 2,
            "Aligned sequences should be at most twice as long as the longest original sequence in {} mode.",
            mode
        );

        let o_sequences = original.items.iter().map(|(_, seq)| seq.clone()).collect::<Vec<_>>();
        let a_sequences = aligned.items.iter().map(|(_, seq)| seq.without_gaps()).collect::<Vec<_>>();

        assert_eq!(o_sequences.len(), a_sequences.len(), "Number of sequences should match in {} mode.", mode);
        assert_eq!(o_sequences, a_sequences, "Sequences should match after alignment in {} mode.", mode);

        for (i, ((o_id, o_seq), (a_id, a_seq))) in original.items.iter().zip(aligned.items.iter()).enumerate() {
            assert_eq!(o_id, a_id, "Sequence IDs at index {} do not match after alignment in {} mode.", i, mode);
            assert_eq!(
                o_seq,
                &a_seq.without_gaps(),
                "Sequence at index {} does not match after removing gaps in {} mode.",
                i,
                mode
            );
        }
    }

    #[test]
    fn test_msa_small() -> Result<(), String> {
        let metric = |a: &String, b: &String| levenshtein::<u8>(a, b);
        let cost_matrix = CostMatrix::<u8>::default();
        let sequences = vec![
            "ACTGA".to_string(),
            "CTGAA".to_string(),
            "TGAAC".to_string(),
            "GAACT".to_string(),
            "AACTG".to_string(),
        ];

        let tree = Tree::new_minimal(sequences.clone(), &metric)?;
        let msa_tree = tree.clone().into_msa(&cost_matrix);
        check_sequences_equal(&tree, &msa_tree, "recursive");

        let msa_tree_iterative = tree.clone().into_msa_iterative(&cost_matrix);
        check_sequences_equal(&tree, &msa_tree_iterative, "iterative");

        let par_tree = Tree::par_new_minimal(sequences, metric)?;
        let par_msa_tree = par_tree.clone().par_into_msa(&cost_matrix);
        check_sequences_equal(&par_tree, &par_msa_tree, "parallel recursive");

        let par_msa_tree_iterative = par_tree.clone().par_into_msa_iterative(&cost_matrix);
        check_sequences_equal(&par_tree, &par_msa_tree_iterative, "parallel iterative");

        Ok(())
    }

    #[test_case(20)]
    #[test_case(50)]
    #[test_case(100)]
    fn test_msa_medium(car: usize) -> Result<(), String> {
        let metric = |a: &String, b: &String| levenshtein::<u16>(a, b);
        let cost_matrix = CostMatrix::<u16>::default();

        let (min_len, max_len) = (8, 12);
        let characters = ['A', 'C', 'G', 'T'];
        let mut rng = rand::rng();
        let sequences = (0..car)
            .map(|_| {
                let len: usize = rng.random_range(min_len..=max_len);
                (0..len).map(|_| characters[rng.random_range(0..characters.len())]).collect::<String>()
            })
            .collect::<Vec<String>>();

        let tree = Tree::new_minimal(sequences.clone(), &metric)?;
        let msa_tree = tree.clone().into_msa(&cost_matrix);
        check_sequences_equal(&tree, &msa_tree, "recursive");

        let msa_tree_iterative = tree.clone().into_msa_iterative(&cost_matrix);
        check_sequences_equal(&tree, &msa_tree_iterative, "iterative");

        let par_tree = Tree::par_new_minimal(sequences, metric)?;
        let par_msa_tree = par_tree.clone().par_into_msa(&cost_matrix);
        check_sequences_equal(&par_tree, &par_msa_tree, "parallel recursive");

        let par_msa_tree_iterative = par_tree.clone().par_into_msa_iterative(&cost_matrix);
        check_sequences_equal(&par_tree, &par_msa_tree_iterative, "parallel iterative");

        Ok(())
    }
}
