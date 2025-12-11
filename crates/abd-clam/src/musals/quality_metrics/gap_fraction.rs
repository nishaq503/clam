//! The mean fraction of gaps in the sequences of a multiple sequence alignment (MSA).

use rayon::prelude::*;

use crate::{
    DistanceValue, Tree,
    musals::{CostMatrix, Sequence},
};

use super::{MsaQuality, mu_sigma_min_max};

/// The mean of the fraction of gaps in the sequences of the MSA.
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct GapFraction {
    /// Mean
    mean: f64,
    /// Standard Deviation
    std_dev: f64,
    /// Minimum
    min: f64,
    /// Maximum
    max: f64,
}

impl MsaQuality for GapFraction {
    fn name(&self) -> String {
        "MeanGapFraction".to_string()
    }

    fn short_name<'a>(&self) -> &'a str {
        "gf"
    }

    fn description(&self) -> String {
        "The mean of the fraction of gaps in each sequence of the MSA.".to_string()
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
        let msa_width = msa_tree.items[0].1.as_ref().len();
        let gap_fractions = msa_tree
            .items
            .iter()
            .map(|(_, seq)| seq.gap_count() as f64 / msa_width as f64)
            .collect::<Vec<_>>();
        let (mean, std_dev, min, max) = mu_sigma_min_max(gap_fractions);
        Self { mean, std_dev, min, max }
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
        let msa_width = msa_tree.items[0].1.as_ref().len();
        let gap_fractions = msa_tree
            .items
            .par_iter()
            .map(|(_, seq)| seq.gap_count() as f64 / msa_width as f64)
            .collect::<Vec<_>>();
        let (mean, std_dev, min, max) = mu_sigma_min_max(gap_fractions);
        Self { mean, std_dev, min, max }
    }
}
