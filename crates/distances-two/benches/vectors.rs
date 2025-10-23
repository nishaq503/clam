//! Benchmarks for vector distances.

#![allow(missing_docs)]

use criterion::*;
use rand::{distr::uniform::SampleUniform, prelude::*};

/// Helper to configure a Criterion benchmark group for distance computations.
pub fn config_group(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>, car: usize) {
    let n_elements = car * car;
    group.throughput(criterion::Throughput::Elements(n_elements as u64));

    group.sample_size(30);

    let plot_config = criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic);
    group.plot_config(plot_config);
}

/// Generates random data for benchmarking distance computations.
pub fn gen_data<F: num_traits::Float + SampleUniform>(
    car: usize,
    dim: usize,
    min_val: F,
    max_val: F,
    seed: u64,
) -> Vec<Vec<F>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    (0..car)
        .map(|_| {
            (0..dim)
                .map(|_| rand::Rng::random_range(&mut rng, min_val..=max_val))
                .collect()
        })
        .collect()
}

/// Helper macro to benchmark one distance function.
macro_rules! bench_one_dist_fn {
    ($id:expr, $dim:expr, $group:expr, $data:expr, $distance_fn:expr) => {
        let id = BenchmarkId::new($id, $dim);
        $group.bench_with_input(id, &$dim, |b, _| {
            b.iter_with_large_drop(|| {
                $data
                    .iter()
                    .map(|x| {
                        $data
                            .iter()
                            .map(|y| std::hint::black_box($distance_fn(x, y)))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            });
        });
    };
}

/// Helper macro to benchmark SIMD distance functions.
macro_rules! bench_simd_fns {
    ($feature:expr, $dim:expr, $group:expr, $data:expr) => {
        #[cfg(feature = $feature)]
        {
            let simd_l2_sq_id = format!("l2_sq_{}", $feature);
            bench_one_dist_fn!(&simd_l2_sq_id, $dim, $group, $data, distances_two::simd::euclidean_sq);

            let simd_l2_id = format!("l2_{}", $feature);
            bench_one_dist_fn!(&simd_l2_id, $dim, $group, $data, distances_two::simd::euclidean);

            let simd_dot_id = format!("dot_{}", $feature);
            bench_one_dist_fn!(&simd_dot_id, $dim, $group, $data, distances_two::simd::dot_product);
        }
    };
}

/// Helper macro to benchmark several distance functions.
macro_rules! bench_many_dist_fns {
    ($seed:expr, $car:expr, $max_dim_pow:expr, $type:ty, $c:expr) => {
        let group_name = concat!("vectors_", stringify!($type));
        let mut group = $c.benchmark_group(group_name);

        config_group(&mut group, $car);

        let dims = (0..=$max_dim_pow)
            .map(|d| 1_000 * 2_u32.pow(d) as usize)
            .collect::<Vec<_>>();

        for &dim in &dims {
            let data = gen_data::<$type>($car, dim, 1.0, 2.0, $seed);
            bench_one_dist_fn!(
                "l2_sq_naive",
                dim,
                &mut group,
                data,
                distances_two::vectors::euclidean_sq
            );
            bench_one_dist_fn!("l2_naive", dim, &mut group, data, distances_two::vectors::euclidean);
            bench_one_dist_fn!("dot_naive", dim, &mut group, data, distances_two::vectors::dot_product);

            #[cfg(feature = "blas")]
            {
                bench_one_dist_fn!("l2_sq_blas", dim, &mut group, data, distances_two::blas::euclidean_sq);
                bench_one_dist_fn!("l2_blas", dim, &mut group, data, distances_two::blas::euclidean);
                bench_one_dist_fn!("dot_blas", dim, &mut group, data, distances_two::blas::dot_product);
            }

            bench_simd_fns!("simd-128", dim, &mut group, data);
            bench_simd_fns!("simd-256", dim, &mut group, data);
            bench_simd_fns!("simd-512", dim, &mut group, data);
            bench_simd_fns!("simd-1024", dim, &mut group, data);
        }

        group.finish();
    };
}

fn vector_distances(c: &mut Criterion) {
    let seed = 42_u64;
    let car = 500; // Number of vectors to compare. The number of distance computations is car^2.
    let max_dim_pow = 0_u32; // Max dimension is 1000 * 2^max_dim_pow.

    bench_many_dist_fns!(seed, car, max_dim_pow, f32, c);
    bench_many_dist_fns!(seed, car, max_dim_pow, f64, c);
}

criterion_group!(benches, vector_distances);
criterion_main!(benches);
