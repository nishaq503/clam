//! Subcommands for MUSALS

use std::path::PathBuf;

use clap::Subcommand;

/// The MUSALS subcommands for building and evaluating MSAs.
#[derive(Subcommand, Debug)]
pub enum MusalsAction {
    /// Build an MSA and save it to a new file.
    Build {
        /// The path to the output directory.
        #[arg(short('o'), long)]
        out_dir: PathBuf,
        /// The cost matrix to use.
        #[arg(short('c'), long, default_value_t = ShellCostMatrix::Default)]
        cost_matrix: ShellCostMatrix,
    },
    /// Evaluate the quality of an MSA.
    Evaluate {
        /// The path to `out_dir` as specified in the `build` command.
        #[arg(short('o'), long)]
        out_dir: PathBuf,
        // TODO Emily: Add or change options as needed
    },
}

#[derive(Debug, Clone, PartialEq, Eq, clap::ValueEnum, Default)]
pub enum ShellCostMatrix {
    /// The default cost matrix, with gap open = 1, gap extend = 1, substitution = 1, and match = 0.
    #[default]
    Default,
    /// As the default, but with affine gap penalties, i.e. gap open = 10.
    DefaultAffine,
    /// The extended IUPAC cost matrix.
    ExtendedIupac,
    /// The BLOSUM62 cost matrix.
    Blosum62,
}

impl core::fmt::Display for ShellCostMatrix {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            ShellCostMatrix::Default => "default",
            ShellCostMatrix::DefaultAffine => "default-affine",
            ShellCostMatrix::ExtendedIupac => "extended-iupac",
            ShellCostMatrix::Blosum62 => "blosum62",
        };
        write!(f, "{s}")
    }
}
