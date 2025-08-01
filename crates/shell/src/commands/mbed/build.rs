//! Build the dimension reduction.

use abd_clam::{
    Ball, Dataset, FlatVec, ParDiskIO,
    cluster::{BalancedBall, ParPartition},
    dataset::{AssociatesMetadata, AssociatesMetadataMut},
    mbed::Complex,
    metric::ParMetric,
};
use distances::{Number, number::Float};
use ndarray::prelude::*;

use crate::{data::ShellFlatVec, metrics::ShellMetric};

/// Build the dimension reduction.
///
/// # Type Parameters
///
/// - `P`: The type of the path to the output directory.
/// - `F`: The type of the floating-point numbers in the reduction.
/// - `DIM`: The number of dimensions.
#[allow(clippy::too_many_arguments)]
pub fn build_new_embedding<P, F, const DIM: usize>(
    out_dir: &P,
    data: &ShellFlatVec,
    metric: &ShellMetric,
    balanced: bool,
    seed: Option<u64>,
    beta: F,
    k: F,
    dk: F,
    dt: F,
    patience: usize,
    target: F,
    max_steps: usize,
) -> Result<FlatVec<[f32; DIM], usize>, String>
where
    P: AsRef<std::path::Path>,
    F: Float,
{
    ftlog::info!("Building the dimension reduction...");
    ftlog::info!("Output directory: {:?}", out_dir.as_ref());
    ftlog::info!("Dimensions: {DIM}");

    match metric {
        ShellMetric::Levenshtein(metric) => match data {
            ShellFlatVec::String(data) => build_generic::<_, _, u32, _, _, _, DIM>(
                out_dir, data, metric, balanced, seed, beta, k, dk, dt, patience, target, max_steps,
            ),
            _ => Err("The Levenshtein metric can only be used with string data.".to_string()),
        },
        ShellMetric::Euclidean(metric) => match data {
            ShellFlatVec::String(_) => Err("The Euclidean metric cannot be used with string data.".to_string()),
            ShellFlatVec::F32(data) => build_generic::<_, _, f32, _, _, _, DIM>(
                out_dir, data, metric, balanced, seed, beta, k, dk, dt, patience, target, max_steps,
            ),
            _ => {
                todo!("Implement remaining match arms")
            }
        },
        ShellMetric::Cosine(metric) => match data {
            ShellFlatVec::String(_) => Err("The Cosine metric cannot be used with string data.".to_string()),
            ShellFlatVec::F32(data) => build_generic::<_, _, f32, _, _, _, DIM>(
                out_dir, data, metric, balanced, seed, beta, k, dk, dt, patience, target, max_steps,
            ),
            _ => {
                todo!("Implement remaining match arms")
            }
        },
    }
}

/// Generic helper for building the dimension reduction.
///
/// # Type Parameters
///
/// - `P`: The type of the path to the output directory.
/// - `I`: The type of the items in the dataset.
/// - `T`: The type of the distance values.
/// - `M`: The type of the distance metric.
/// - `Me`: The type of the metadata with the dataset.
/// - `F`: The type of the floating-point numbers in the reduction.
/// - `DIM`: The number of dimensions.
#[allow(clippy::too_many_arguments)]
fn build_generic<P, I, T, M, Me, F, const DIM: usize>(
    out_dir: &P,
    data: &FlatVec<I, Me>,
    metric: &M,
    balanced: bool,
    seed: Option<u64>,
    beta: F,
    k: F,
    dk: F,
    dt: F,
    patience: usize,
    target: F,
    max_steps: usize,
) -> Result<FlatVec<[f32; DIM], Me>, String>
where
    P: AsRef<std::path::Path>,
    I: Send + Sync,
    T: Number + bitcode::Encode + bitcode::Decode,
    M: ParMetric<I, T>,
    Me: Clone + Send + Sync,
    F: Float,
{
    let mut rng = rand::rng();

    ftlog::info!("Creating the tree...");
    let tree_path = out_dir.as_ref().join(format!("{}-tree.bin", data.name()));
    let root = if tree_path.exists() {
        Ball::<T>::par_read_from(&tree_path)?
    } else {
        let root = if balanced {
            BalancedBall::par_new_tree_iterative(data, metric, &|_| true, seed, 128).into_ball()
        } else {
            Ball::par_new_tree_iterative(data, metric, &|_| true, seed, 128)
        };
        root.par_write_to(&tree_path)?;
        root
    };

    ftlog::info!("Setting up the simulation...");
    let drag_coefficient = F::ONE - beta;
    let spring_constant = k;
    let loosening_factor = dk;
    let mut system = Complex::<_, _, F, DIM>::new(&root, drag_coefficient, spring_constant, loosening_factor);

    ftlog::info!("Running the simulation...");
    let tolerance = target;
    let n = patience;
    let checkpoints = system.par_simulate_to_leaves(&mut rng, data, metric, max_steps, tolerance, dt, n);

    ftlog::info!("Writing the energy history...");
    let energy_history = system
        .energy_history()
        .iter()
        .map(|&(ke, pe)| [ke.as_f32(), pe.as_f32()])
        .collect::<Vec<_>>();
    let energy_history =
        FlatVec::new(energy_history).unwrap_or_else(|_| unreachable!("The simulation took some steps!"));
    let energy_history = energy_history.to_array2();
    let energy_path = out_dir.as_ref().join(format!("{}-energy.npy", data.name()));
    ndarray_npy::write_npy(&energy_path, &energy_history).map_err(|e| e.to_string())?;

    ftlog::info!("Writing the {} checkpoints...", checkpoints.len());
    let steps = checkpoints
        .into_iter()
        .map(|step| {
            step.transform_items(|row| {
                let mut ret = [0.0; DIM];
                for (a, b) in ret.iter_mut().zip(row.iter()) {
                    *a = b.as_f32();
                }
                ret
            })
        })
        .collect::<Vec<_>>();
    let final_step = steps
        .last()
        .unwrap_or_else(|| unreachable!("We have performed at least one step."))
        .clone();

    let arrays = steps.into_iter().map(|step| step.to_array2()).collect::<Vec<_>>();
    let arrays = arrays.iter().map(ArrayBase::view).collect::<Vec<_>>();
    let stack = ndarray::stack(ndarray::Axis(0), &arrays).map_err(|e| e.to_string())?;
    let stack_path = out_dir.as_ref().join(format!("{}-stack.npy", data.name()));
    ndarray_npy::write_npy(&stack_path, &stack).map_err(|e| e.to_string())?;

    ftlog::info!("Returning the resulting embedding...");
    final_step.with_metadata(data.metadata())
}
