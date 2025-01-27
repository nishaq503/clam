//! Steps for the MSA pipeline.

use std::path::Path;

use abd_clam::{
    cakes::PermutedBall,
    cluster::{adapter::ParBallAdapter, BalancedBall, ClusterIO, Csv, ParPartition},
    dataset::{AssociatesMetadata, AssociatesMetadataMut, DatasetIO},
    metric::ParMetric,
    musals::{Aligner, Columns, Sequence, MSA},
    Ball, Cluster, Dataset, FlatVec,
};

type B<U> = Ball<U>;
type Pb<U> = PermutedBall<U, B<U>>;

/// Build the aligned datasets.
pub fn build_aligned<P: AsRef<Path>>(
    matrix: &crate::CostMatrix,
    gap_open: Option<usize>,
    perm_ball: &Pb<i32>,
    data: &FlatVec<Sequence<i32>, String>,
    out_path: &P,
) -> Result<(), String> {
    ftlog::info!("Setting up aligner...");
    let gap = b'-';
    let cost_matrix = matrix.cost_matrix::<i32>(gap_open);
    let aligner = Aligner::new(&cost_matrix, gap);

    ftlog::info!("Aligning sequences...");
    let builder = Columns::new(gap).par_with_tree(perm_ball, data, &aligner);

    ftlog::info!("Extracting aligned sequences...");
    let msa = builder.to_flat_vec_rows().with_metadata(data.metadata())?;
    let transformer = |s: Vec<u8>| s.into_iter().map(|c| c as char).collect::<String>();
    let msa = msa.transform_items(transformer);

    ftlog::info!("Finished aligning {} sequences.", builder.len());
    let data = MSA::new(&aligner, msa)?;

    ftlog::info!("Writing MSA to {:?}", out_path.as_ref());
    bench_utils::fasta::write(&data, out_path)?;

    Ok(())
}

/// Read the aligned fasta file.
pub fn read_aligned<P: AsRef<Path>>(path: &P, aligner: &Aligner<i32>) -> Result<MSA<String, i32, String>, String> {
    ftlog::info!("Reading aligned sequences from {:?}", path.as_ref());
    let (data, _) = bench_utils::fasta::read(path, 0, false)?;
    MSA::new(aligner, data)
}

/// Build the `PermutedBall` and the permuted dataset.
#[allow(clippy::type_complexity)]
pub fn build_perm_ball<'a, P: AsRef<Path>, M: ParMetric<Sequence<'a, i32>, i32>>(
    ball: B<i32>,
    data: FlatVec<Sequence<'a, i32>, String>,
    metric: &M,
    ball_path: &P,
    data_path: &P,
) -> Result<(Pb<i32>, FlatVec<Sequence<'a, i32>, String>), String> {
    ftlog::info!("Building PermutedBall and permuted dataset.");
    let (ball, data) = PermutedBall::par_from_ball_tree(ball, data, metric);

    ftlog::info!("Writing PermutedBall to {:?}", ball_path.as_ref());
    ball.write_to(ball_path)?;

    ftlog::info!("Writing PermutedData to {:?}", data_path.as_ref());
    let transformer = |seq: Sequence<'a, i32>| seq.seq().to_string();
    let writable_data = data.clone().transform_items(transformer);
    writable_data.write_to(data_path)?;

    Ok((ball, data))
}

/// Read the `PermutedBall` and the permuted dataset from disk.
#[allow(clippy::type_complexity)]
pub fn read_permuted_ball<'a, P: AsRef<Path>>(
    ball_path: &P,
    data_path: &P,
    aligner: &'a Aligner<i32>,
) -> Result<(Pb<i32>, FlatVec<Sequence<'a, i32>, String>), String> {
    ftlog::info!("Reading PermutedBall from {:?}", ball_path.as_ref());
    let ball = Pb::read_from(ball_path)?;

    ftlog::info!("Reading PermutedData from {:?}", data_path.as_ref());
    let data = FlatVec::<String, String>::read_from(data_path)?;
    let transformer = |s: String| Sequence::new(s, Some(aligner));
    let data = data.transform_items(transformer);

    Ok((ball, data))
}

/// Build the Ball and the dataset.
pub fn build_ball<'a, P: AsRef<Path>, M: ParMetric<Sequence<'a, i32>, i32>>(
    data: &FlatVec<Sequence<'a, i32>, String>,
    metric: &M,
    ball_path: &P,
    csv_path: &P,
    balanced: bool,
) -> Result<B<i32>, String> {
    // Create the ball from scratch.
    let seed = Some(42);
    let depth_stride = abd_clam::utils::max_recursion_depth();
    let ball = if balanced {
        ftlog::info!("Building BalancedBall on dataset with {} items.", data.cardinality());
        BalancedBall::par_new_tree_iterative(data, metric, &|_| true, seed, depth_stride).into_ball()
    } else {
        ftlog::info!("Building Ball on dataset with {} items.", data.cardinality());
        Ball::par_new_tree_iterative(data, metric, &|_| true, seed, depth_stride)
    };

    let num_leaves = ball.leaves().len();
    ftlog::info!("Built Ball with {num_leaves} leaves.");

    // Serialize the ball to disk.
    ftlog::info!("Writing Ball to {:?}", ball_path.as_ref());
    ball.write_to(ball_path)?;

    // Write the ball to a CSV file.;
    ftlog::info!("Writing Ball to CSV at {:?}", csv_path.as_ref());
    ball.write_to_csv(&csv_path)?;

    Ok(ball)
}

/// Read the Ball from disk.
pub fn read_ball<P: AsRef<Path>>(path: &P) -> Result<B<i32>, String> {
    ftlog::info!("Reading ball from {:?}", path.as_ref());
    let ball = Ball::read_from(path)?;
    ftlog::info!("Finished reading Ball.");

    Ok(ball)
}
