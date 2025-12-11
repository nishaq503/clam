//! CLI for CLAM-MBED, the dimension reduction tool.

mod commands;
pub mod data;
pub mod metrics;
mod search;
mod trees;
pub mod utils;

use std::path::PathBuf;

use clap::Parser;
use rand::prelude::*;

use commands::Commands;

use crate::{data::InputFormat, metrics::Metric};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The path to the input dataset file (not required for generate-data).
    #[arg(short('i'), long)]
    inp_path: Option<PathBuf>,

    /// The path to the output directory or file, depending on the subcommand used.
    #[arg(short('o'), long)]
    out_path: PathBuf,

    /// The name of the distance metric to use for the CAKES tree.
    #[arg(short('m'), long)]
    metric: Metric,

    /// The random seed to use.
    #[arg(short('s'), long)]
    seed: Option<u64>,

    /// Whether to use the iterative version of the partition algorithm in order to avoid stack overflows from deep recursion. If true, the maximum recursion
    /// depth is limited to 128.
    #[arg(long, default_value_t = false)]
    iterative_partition: bool,

    /// Optional size of the subsample of the input data to use for building the tree.
    #[arg(short('n'), long)]
    sample_size: Option<usize>,

    /// The name of the log-file to use.
    #[arg(short('l'), long, default_value = "shell.log")]
    log_name: String,

    /// The subcommand to run.
    #[command(subcommand)]
    command: Commands,
}

fn main() -> Result<(), String> {
    let args = Args::parse();

    let out_path = &args.out_path;

    let (_guard, log_path) = utils::configure_logger(&args.log_name)?;
    println!("Log file: {log_path:?}");

    let inp_path = args
        .inp_path
        .ok_or_else(|| "Input path (-i/--inp-path) is required for this command".to_string())?;
    let metric = args.metric;
    let sample_size = args.sample_size;
    let mut rng = args.seed.map_or_else(rand::rngs::StdRng::from_os_rng, rand::rngs::StdRng::seed_from_u64);

    match args.command {
        Commands::GenerateData { action } => match action {
            commands::generate_data::GenerateDataAction::Generate {
                num_vectors,
                dimensions,
                data_type,
                partitions,
                min_val,
                max_val,
            } => commands::generate_data::generate_dataset(num_vectors, dimensions, out_path, data_type, partitions, min_val, max_val, &mut rng),
        },
        Commands::Cakes { action } => match action {
            commands::cakes::CakesAction::Build => {
                let inp_data = InputFormat::read(&inp_path, sample_size, &mut rng)?;
                commands::cakes::build_new_tree(inp_data, &metric, args.iterative_partition, out_path)
            }
            commands::cakes::CakesAction::Search {
                queries_path,
                cakes_algorithms,
            } => {
                let queries = InputFormat::read(&queries_path, sample_size, &mut rng)?;
                commands::cakes::search_tree(&inp_path, queries, &cakes_algorithms, out_path)
            }
        },
        Commands::Musals { action, cost_matrix } => match action {
            commands::musals::MusalsAction::Build { save_fasta } => {
                let inp_data = InputFormat::read(&inp_path, sample_size, &mut rng)?;
                commands::musals::build_msa(inp_data, &metric, args.iterative_partition, &cost_matrix, out_path, save_fasta)
            }
            commands::musals::MusalsAction::Evaluate { quality_metrics } => commands::musals::evaluate_msa(&inp_path, &quality_metrics, &cost_matrix, out_path),
        },
        // Commands::Mbed { action, .. } => { todo!() }
    }
}
