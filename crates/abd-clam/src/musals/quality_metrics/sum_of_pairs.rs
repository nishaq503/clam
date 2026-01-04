//! Scores each pairwise alignment in the MSA, applying a penalty for gaps and mismatches.

use rayon::prelude::*;

use crate::{
    DistanceValue, Tree,
    musals::{CostMatrix, Sequence},
};

use super::{MsaQuality, mu_sigma_min_max, random_sample_indices};

/// The Sum of Pairs (SP) score of the MSA.
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct SumOfPairs {
    /// Mean
    mean: f64,
    /// Standard Deviation
    std_dev: f64,
    /// Minimum
    min: f64,
    /// Maximum
    max: f64,
}

impl MsaQuality for SumOfPairs {
    fn name(&self) -> String {
        "SumOfPairs".to_string()
    }

    fn short_name<'a>(&self) -> &'a str {
        "sop"
    }

    fn description(&self) -> String {
        "The Sum of Pairs (SP) score of the MSA.".to_string()
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

    fn compute<Id, S, T, A, M>(msa_tree: &Tree<Id, S, T, A, M>, cost_matrix: &CostMatrix<T>, sample_size: Option<usize>) -> Self
    where
        S: Sequence,
        T: DistanceValue,
        M: Fn(&S, &S) -> T,
        Self: Sized,
    {
        let indices = random_sample_indices(msa_tree.cardinality(), sample_size);
        let items = indices.iter().map(|&i| &msa_tree.items[i]).collect::<Vec<_>>();
        let columns = msa_to_columns(&items);
        let scores = columns.iter().map(|col| column_score(col, cost_matrix)).collect::<Vec<_>>();
        let (mean, std_dev, min, max) = mu_sigma_min_max(&scores);
        Self { mean, std_dev, min, max }
    }

    fn par_compute<Id, S, T, A, M>(msa_tree: &Tree<Id, S, T, A, M>, cost_matrix: &CostMatrix<T>, sample_size: Option<usize>) -> Self
    where
        Id: Send + Sync,
        S: Sequence + Send + Sync,
        T: DistanceValue + Send + Sync,
        A: Send + Sync,
        M: Fn(&S, &S) -> T + Send + Sync,
        Self: Sized + Send + Sync,
    {
        let indices = random_sample_indices(msa_tree.cardinality(), sample_size);
        let items = indices.iter().map(|&i| &msa_tree.items[i]).collect::<Vec<_>>();
        let columns = par_msa_to_columns(&items);
        let scores = columns.par_iter().map(|col| column_score(col, cost_matrix)).collect::<Vec<_>>();
        let (mean, std_dev, min, max) = mu_sigma_min_max(&scores);
        Self { mean, std_dev, min, max }
    }
}

/// Converts the MSA from rows to columns.
fn msa_to_columns<Id, S>(msa: &[&(Id, S)]) -> Vec<S>
where
    S: Sequence,
{
    let seqs = msa.iter().map(|(_, seq)| seq.as_ref()).collect::<Vec<_>>();
    let n_cols = seqs[0].len();
    (0..n_cols).map(|col_idx| S::from_vec(seqs.iter().map(|seq| seq[col_idx]).collect())).collect()
}

/// Parallel version of [`msa_to_columns`].
fn par_msa_to_columns<Id, S>(msa: &[&(Id, S)]) -> Vec<S>
where
    S: Sequence + Send + Sync,
{
    let seqs = msa.iter().map(|(_, seq)| seq.as_ref()).collect::<Vec<_>>();
    let n_cols = seqs[0].len();
    (0..n_cols)
        .into_par_iter()
        .map(|col_idx| S::from_vec(seqs.iter().map(|seq| seq[col_idx]).collect()))
        .collect()
}

/// Computes the Sum of Pairs score for a single column in the MSA, normalized by the number of pairs.
fn column_score<S, T>(column: &S, cost_matrix: &CostMatrix<T>) -> f64
where
    S: Sequence,
    T: DistanceValue,
{
    let frequencies = count_pairs(column.as_ref());
    let total_pairs: usize = frequencies.iter().sum();

    let score = frequencies.iter().enumerate().filter(|&(_, &count)| count > 0).fold(0_f64, |acc, (k, &count)| {
        let (i, j) = flat_to_lt_index(k);
        let pair_score = if i == S::GAP || j == S::GAP {
            cost_matrix.gap_ext_cost()
        } else {
            cost_matrix.sub_cost(i, j)
        }
        .to_f64()
        .unwrap_or_else(|| unreachable!("Failed to convert DistanceValue to f64"));
        pair_score.mul_add(count as f64, acc)
    });

    score / (total_pairs as f64)
}

/// Counts the frequency of each pair of different characters in the column.
fn count_pairs(column: &[u8]) -> Vec<usize> {
    let mut pair_counts = vec![0; 255 * 128]; // Assuming ASCII characters

    for (i, &a) in column.iter().enumerate() {
        for &b in &column[i + 1..] {
            let (i, j) = if a < b { (a, b) } else { (b, a) };
            pair_counts[lt_to_flat_index(i, j)] += 1;
        }
    }

    pair_counts
}

/// Converts a pair of indices for a lower-triangular matrix into a single index in a flat array.
///
/// This assumes that `i >= j`.
const fn lt_to_flat_index(i: u8, j: u8) -> usize {
    let i = i as usize;
    let j = j as usize;
    (i * (i + 1)) / 2 + j
}

/// Converts a flat array index into a pair of indices for a lower-triangular matrix.
#[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::many_single_char_names)]
fn flat_to_lt_index(k: usize) -> (u8, u8) {
    let p = ((1 + 8 * k) as f64).sqrt();
    let i = ((p - 1.0) / 2.0).floor() as usize;

    let t = i * (i + 1) / 2;
    let j = k - t;

    (i as u8, j as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lt_to_flat_index() {
        assert_eq!(lt_to_flat_index(0, 0), 0);
        assert_eq!(lt_to_flat_index(1, 0), 1);
        assert_eq!(lt_to_flat_index(1, 1), 2);
        assert_eq!(lt_to_flat_index(2, 0), 3);
        assert_eq!(lt_to_flat_index(2, 1), 4);
        assert_eq!(lt_to_flat_index(2, 2), 5);

        assert_eq!(flat_to_lt_index(0), (0, 0));
        assert_eq!(flat_to_lt_index(1), (1, 0));
        assert_eq!(flat_to_lt_index(2), (1, 1));
        assert_eq!(flat_to_lt_index(3), (2, 0));
        assert_eq!(flat_to_lt_index(4), (2, 1));
        assert_eq!(flat_to_lt_index(5), (2, 2));

        let mut k = 0;
        for i in 0..=255 {
            for j in 0..=i {
                assert_eq!(lt_to_flat_index(i, j), k);

                let (a, b) = flat_to_lt_index(k);
                assert_eq!((i, j), (a, b), "Failed at k = {k}");

                k += 1;
            }
        }
    }
}
