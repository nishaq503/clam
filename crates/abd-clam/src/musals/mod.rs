//! Multiple Sequence Alignment At Scale (`MuSAlS`) with CLAM.

mod alignment_ops;
mod columnar;
mod cost_matrix;
mod msa;
mod sequence;

pub use cost_matrix::CostMatrix;
pub use sequence::Sequence;

use alignment_ops::{Direction, Edit, Edits};
use columnar::Columnar;
use msa::Msa;

use rayon::prelude::*;

use crate::{DistanceValue, Tree};

impl<Id, S, T, A, M> Tree<Id, S, T, A, M>
where
    S: Sequence,
    T: DistanceValue,
{
    /// Returns a new tree containing the multiple sequence alignment of the sequences in the original tree.
    pub fn into_msa(mut self, cost_matrix: &CostMatrix<T>) -> Self {
        let msa = Msa::from_tree(&self, cost_matrix);

        self.items = self
            .items
            .into_iter()
            .zip(msa)
            .map(|((id, _), aligned_seq)| (id, aligned_seq))
            .collect();

        self
    }
}

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
        let msa = Msa::par_from_tree(&self, cost_matrix);

        self.items = self
            .items
            .into_par_iter()
            .zip(msa)
            .map(|((id, _), aligned_seq)| (id, aligned_seq))
            .collect();

        self
    }
}

#[cfg(test)]
mod tests {
    use distances::strings::levenshtein;
    use rand::prelude::*;
    use test_case::test_case;

    use crate::Tree;

    use super::{CostMatrix, Sequence};

    fn check_sequences_equal<Id, S, T, A, M>(original: Tree<Id, S, T, A, M>, aligned: Tree<Id, S, T, A, M>)
    where
        Id: Eq + core::fmt::Debug,
        S: Sequence + Eq + core::fmt::Debug,
    {
        assert_eq!(
            original.cardinality(),
            aligned.cardinality(),
            "Number of sequences should match."
        );

        let max_len = original
            .items
            .iter()
            .map(|(_, seq)| seq.as_ref().len())
            .max()
            .unwrap_or(0);
        let aligned_max_len = aligned
            .items
            .iter()
            .map(|(_, seq)| seq.as_ref().len())
            .max()
            .unwrap_or(0);
        assert!(
            aligned_max_len >= max_len,
            "Aligned sequences should be at least as long as the longest original sequence."
        );
        assert!(
            aligned_max_len <= max_len * 2,
            "Aligned sequences should be at most twice as long as the longest original sequence."
        );

        for (i, ((o_id, o_seq), (a_id, a_seq))) in original.items.into_iter().zip(aligned.items).enumerate() {
            assert_eq!(o_id, a_id, "Sequence IDs at index {} do not match after alignment.", i);

            assert_eq!(
                o_seq,
                a_seq.without_gaps(),
                "Sequence at index {} does not match after removing gaps.",
                i
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
        check_sequences_equal(tree, msa_tree);

        let par_tree = Tree::par_new_minimal(sequences, metric)?;
        let par_msa_tree = par_tree.clone().par_into_msa(&cost_matrix);
        check_sequences_equal(par_tree, par_msa_tree);

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
                (0..len)
                    .map(|_| characters[rng.random_range(0..characters.len())])
                    .collect::<String>()
            })
            .collect::<Vec<String>>();

        let tree = Tree::new_minimal(sequences.clone(), &metric)?;
        let msa_tree = tree.clone().into_msa(&cost_matrix);
        check_sequences_equal(tree, msa_tree);

        let par_tree = Tree::par_new_minimal(sequences, metric)?;
        let par_msa_tree = par_tree.clone().par_into_msa(&cost_matrix);
        check_sequences_equal(par_tree, par_msa_tree);

        Ok(())
    }
}
