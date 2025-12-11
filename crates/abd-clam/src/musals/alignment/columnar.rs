//! Multiple sequences of the same length stored in a columnar format.

use rayon::prelude::*;

use crate::DistanceValue;

use super::{CostMatrix, Sequence};

/// A multiple sequence alignment stored in a columnar format.
#[derive(Clone, Debug)]
pub struct Columnar<S: Sequence>(Vec<S>);

impl<S: Sequence> core::ops::Deref for Columnar<S> {
    type Target = Vec<S>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<S: Sequence> FromIterator<S> for Columnar<S> {
    fn from_iter<I: IntoIterator<Item = S>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<S: Sequence + Send + Sync> FromParallelIterator<S> for Columnar<S> {
    fn from_par_iter<I: IntoParallelIterator<Item = S>>(par_iter: I) -> Self {
        Self(par_iter.into_par_iter().collect())
    }
}

impl<S: Sequence> IntoIterator for Columnar<S> {
    type Item = S;
    type IntoIter = std::vec::IntoIter<S>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<S: Sequence + Send + Sync> IntoParallelIterator for Columnar<S> {
    type Item = S;
    type Iter = rayon::vec::IntoIter<S>;

    fn into_par_iter(self) -> Self::Iter {
        self.0.into_par_iter()
    }
}

/// A trait for types that can be represented in a columnar format.
impl<S: Sequence> Columnar<S> {
    /// The number of sequences.
    fn n_seq(&self) -> usize {
        self[0].as_ref().len()
    }

    /// The length of each sequence.
    fn width(&self) -> usize {
        self.len()
    }

    /// Returns the last row as a representative sequence.
    fn last_row(&self) -> S {
        let h = self.n_seq() - 1;
        let mut last = vec![S::GAP; self.width()];
        for (c, col) in self.iter().enumerate() {
            last[c] = col.as_ref()[h];
        }
        S::from_vec(last)
    }

    /// Creates a new columnar structure from a single row.
    pub fn from_row(row: &S) -> Self {
        row.as_ref().iter().map(|&byte| S::splat(byte, 1)).collect()
    }

    /// Converts the columnar structure back to a vector of sequences as rows.
    pub fn into_rows(self, reverse: bool) -> Vec<S> {
        let n_seq = self.n_seq();
        let width = self.width();

        let mut rows = vec![vec![S::GAP; width]; n_seq];
        for (c, col) in self.into_iter().enumerate() {
            for (i, &byte) in col.as_ref().iter().enumerate() {
                rows[i][c] = byte;
            }
        }

        if reverse {
            rows.into_iter().rev().map(S::from_vec).collect()
        } else {
            rows.into_iter().map(S::from_vec).collect()
        }
    }

    /// Returns a column of all gaps.
    fn gap_column(&self) -> S {
        S::splat(S::GAP, self.n_seq())
    }

    /// Inserts gap columns at the specified positions.
    fn with_gaps(self, indices: &[usize]) -> Self {
        let gap_column = self.gap_column();

        let mut cols = self.into_iter().collect::<Vec<S>>();
        for &idx in indices.iter().rev() {
            cols.insert(idx, gap_column.clone());
        }

        Self::from_iter(cols)
    }

    /// Merges another columnar structure into this one.
    pub fn merge<T: DistanceValue>(self, top: Self, cost_matrix: &CostMatrix<T>) -> Self {
        let [b, t] = [self.last_row(), top.last_row()];
        let dp_table = b.nw_table(&t, cost_matrix);
        let [b_gaps, t_gaps] = b.gap_indices(&t, &dp_table);

        let top = Self::from_iter(top).with_gaps(&t_gaps);
        let bottom = Self::from_iter(self).with_gaps(&b_gaps);

        bottom.into_iter().zip(top).map(|(b_col, t_col)| b_col.append(t_col)).collect()
    }

    /// Appends a row to the columnar structure.
    pub fn post_pend_row<T: DistanceValue>(self, row: &S, cost_matrix: &CostMatrix<T>) -> Self {
        let last = self.last_row();
        let dp_table = last.nw_table(row, cost_matrix);
        let [last_gaps, row_gaps] = last.gap_indices(row, &dp_table);
        let row = row.insert_gaps(&row_gaps);

        self.with_gaps(&last_gaps)
            .into_iter()
            .zip(row.as_ref())
            .map(|(col, &byte)| col.post_pend(byte))
            .collect()
    }
}

impl<S: Sequence + Send + Sync> Columnar<S> {
    /// Parallel version of [`Self::into_rows`].
    pub fn par_into_rows(self, reverse: bool) -> Vec<S> {
        let n_seq = self.n_seq();
        let width = self.width();

        let rows = vec![vec![S::GAP; width]; n_seq];
        self.into_par_iter().enumerate().for_each(|(c, col)| {
            col.as_ref().par_iter().enumerate().for_each(|(i, &byte)| {
                // SAFETY: We have exclusive access to each cell in the rows matrix
                // because every (c, i) pair is unique.
                #[allow(unsafe_code)]
                unsafe {
                    let row_ptr = &mut *rows.as_ptr().cast_mut().add(i);
                    row_ptr[c] = byte;
                }
            });
        });

        if reverse {
            rows.into_iter().rev().map(S::from_vec).collect()
        } else {
            rows.into_iter().map(S::from_vec).collect()
        }
    }

    /// Parallel version of [`Self::merge`].
    pub fn par_merge<T: DistanceValue + Send + Sync>(self, top: Self, cost_matrix: &CostMatrix<T>) -> Self {
        let [b, t] = [self.last_row(), top.last_row()];
        let dp_table = b.nw_table(&t, cost_matrix);
        let [b_gaps, t_gaps] = b.gap_indices(&t, &dp_table);

        let top = Self::from_par_iter(top).with_gaps(&t_gaps);
        let bottom = Self::from_par_iter(self).with_gaps(&b_gaps);

        bottom
            .into_par_iter()
            .zip(top.into_par_iter())
            .map(|(b_col, t_col)| b_col.append(t_col))
            .collect()
    }
}
