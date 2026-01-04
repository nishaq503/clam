//! Subcommands for MUSALS

use abd_clam::{
    DistanceValue,
    musals::{CostMatrix, QualityMetric},
};
use clap::Subcommand;

mod build;
mod evaluate;

pub use build::build_msa;
pub use evaluate::evaluate_msa;

/// The MUSALS subcommands for building and evaluating MSAs.
#[derive(Subcommand, Debug)]
pub enum MusalsAction {
    /// Build an MSA and save it to a new file. `out_path` must be a directory.
    Build {
        /// Whether to also save the MSA as a FASTA file. By default, only the binary format is saved.
        #[arg(short('f'), long, default_value_t = false)]
        save_fasta: bool,
    },
    /// Evaluate the quality of an MSA. `out_path` must be a file with '.json' or '.yaml' extension.
    Evaluate {
        /// The quality metrics to compute.
        #[arg(short('q'), long, value_parser = clap::value_parser!(ShellQualityMetric))]
        quality_metrics: Vec<ShellQualityMetric>,
        /// Optional size of the subsample of the input data to use for evaluation. If not provided, the full MSA is used.
        #[arg(short('n'), long)]
        sample_size: Option<usize>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, clap::ValueEnum, Default)]
pub enum ShellCostMatrix {
    /// The default cost matrix, with gap open = 1, gap extend = 1, substitution = 1, and match = 0.
    #[default]
    Default,
    /// As the default, but with affine gap penalties, i.e. gap open = 10.
    Affine,
    /// The extended IUPAC cost matrix.
    ExtendedIupac,
    /// The BLOSUM62 cost matrix.
    Blosum62,
}

impl core::fmt::Display for ShellCostMatrix {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            ShellCostMatrix::Default => "default",
            ShellCostMatrix::Affine => "affine",
            ShellCostMatrix::ExtendedIupac => "extended-iupac",
            ShellCostMatrix::Blosum62 => "blosum62",
        };
        write!(f, "{s}")
    }
}

impl ShellCostMatrix {
    /// Get the matrix for use in MUSALS.
    pub fn get<T: DistanceValue>(&self) -> CostMatrix<T> {
        let mat: CostMatrix<i64> = match self {
            ShellCostMatrix::Default => CostMatrix::default(),
            ShellCostMatrix::Affine => CostMatrix::default_affine(None),
            ShellCostMatrix::ExtendedIupac => CostMatrix::extended_iupac(None),
            ShellCostMatrix::Blosum62 => CostMatrix::blosum62(None),
        };
        let caster = |x: i64| {
            T::from_i64(x).unwrap_or_else(|| {
                panic!(
                    "Cannot convert cost matrix value {x} to target distance value type: {}",
                    std::any::type_name::<T>()
                )
            })
        };
        mat.cast(caster)
    }
}

/// The quality metrics that can be computed for an MSA.
#[derive(Debug, Clone, PartialEq, Eq, clap::ValueEnum)]
pub enum ShellQualityMetric {
    /// The mean fraction of gaps in the sequences of the MSA.
    GapFraction,
    /// The fraction of mismatches between pairs of sequences in the MSA.
    MismatchFraction,
    /// The scores of pairwise alignments in the MSA.
    PairwiseScores,
    /// The scores of weighted pairwise alignments in the MSA.
    WeightedPairwiseScores,
    /// The mean distortion of alignment distances between pairs of sequences in the MSA.
    DistanceDistortion,
    /// The Sum of Pairs (SP) score of the MSA.
    SumOfPairs,
}

impl ShellQualityMetric {
    /// Get the quality metric for use in MUSALS.
    pub fn get(&self) -> QualityMetric {
        match self {
            Self::GapFraction => QualityMetric::GapFraction,
            Self::MismatchFraction => QualityMetric::MismatchFraction,
            Self::PairwiseScores => QualityMetric::PairwiseScores,
            Self::WeightedPairwiseScores => QualityMetric::WeightedPairwiseScores,
            Self::DistanceDistortion => QualityMetric::DistanceDistortion,
            Self::SumOfPairs => QualityMetric::SumOfPairs,
        }
    }

    /// Get the name of the quality metric.
    pub fn name(&self) -> &'static str {
        match self {
            Self::GapFraction => "gap-fraction",
            Self::MismatchFraction => "mismatch-fraction",
            Self::PairwiseScores => "pairwise-scores",
            Self::WeightedPairwiseScores => "weighted-pairwise-scores",
            Self::DistanceDistortion => "distance-distortion",
            Self::SumOfPairs => "sum-of-pairs",
        }
    }
}
