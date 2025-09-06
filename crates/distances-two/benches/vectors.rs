//! Benchmarks for vector distances.

#![allow(missing_docs)]

use criterion::*;
use rand::{distr::uniform::SampleUniform, prelude::*};

/// Helper to configure a Criterion benchmark group for distance computations.
pub fn config_group(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>, car: usize) {
    let n_elements = car * car;
    group.throughput(criterion::Throughput::Elements(n_elements as u64));

    group.sample_size(10);

    let plot_config = criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic);
    group.plot_config(plot_config);
}

/// Generates random data for benchmarking distance computations.
pub fn gen_data<F: num_traits::Float + SampleUniform>(car: usize, dim: usize, min_val: F, max_val: F, seed: u64) -> Vec<Vec<F>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    (0..car)
        .map(|_| (0..dim).map(|_| rand::Rng::random_range(&mut rng, min_val..=max_val)).collect())
        .collect()
}

/// Helper macro to benchmark a distance function between two vectors.
macro_rules! bench_dual_dist_fn {
    ($id:expr, $dim:expr, $group:expr, $data:expr, $distance_fn:expr) => {
        let id = BenchmarkId::new($id, $dim);
        $group.bench_with_input(id, &$dim, |b, _| {
            b.iter_with_large_drop(|| {
                $data
                    .iter()
                    .map(|x| $data.iter().map(|y| std::hint::black_box($distance_fn(x, y))).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            });
        });
    };
}

/// Helper macro to benchmark a distance function for a single vector.
macro_rules! bench_self_dist_fn {
    ($id:expr, $dim:expr, $group:expr, $data:expr, $distance_fn:expr) => {
        let car = $data.len();
        $group.throughput(criterion::Throughput::Elements(car as u64));

        let id = BenchmarkId::new($id, $dim);
        $group.bench_with_input(id, &$dim, |b, _| {
            b.iter_with_large_drop(|| $data.iter().map(|x| std::hint::black_box($distance_fn(x))).collect::<Vec<_>>());
        });

        $group.throughput(criterion::Throughput::Elements((car * car) as u64));
    };
}

/// Helper macro to benchmark SIMD distance functions.
macro_rules! bench_simd_fns {
    ($feature:expr, $dim:expr, $group:expr, $data:expr) => {{
        let simd_l2_id = format!("l2_{}", $feature);
        bench_dual_dist_fn!(&simd_l2_id, $dim, $group, $data, distances_two::simd::euclidean);

        let simd_dot_id = format!("dot_{}", $feature);
        bench_dual_dist_fn!(&simd_dot_id, $dim, $group, $data, distances_two::simd::dot_product);

        let simd_cosine_id = format!("cosine_{}", $feature);
        bench_dual_dist_fn!(&simd_cosine_id, $dim, $group, $data, distances_two::simd::cosine);

        let simd_norm_id = format!("norm_{}", $feature);
        bench_self_dist_fn!(&simd_norm_id, $dim, $group, $data, distances_two::simd::norm_l2);
    }};
}

/// Helper macro to benchmark a distance function between two vectors.
macro_rules! bench_dual_dist_lanes_fn {
    ($id:expr, $dim:expr, $group:expr, $data:expr, $distance_fn:ident, $lanes:expr) => {
        let id = BenchmarkId::new($id, $dim);
        $group.bench_with_input(id, &$dim, |b, _| {
            b.iter_with_large_drop(|| {
                $data
                    .iter()
                    .map(|x| {
                        $data
                            .iter()
                            .map(|y| std::hint::black_box(distances_two::std_simd::$distance_fn::<_, _, $lanes>(x, y)))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            });
        });
    };
}

/// Helper macro to benchmark SIMD distance functions.
macro_rules! bench_simd_lanes_fns {
    ($feature:expr, $id:expr, $dim:expr, $group:expr, $data:expr, $distance_fn:ident) => {{
        // Lane-specific benchmarks.
        bench_dual_dist_lanes_fn!(&format!("{}_lanes_4", $id), $dim, $group, $data, $distance_fn, 4);
        bench_dual_dist_lanes_fn!(&format!("{}_lanes_8", $id), $dim, $group, $data, $distance_fn, 8);
        bench_dual_dist_lanes_fn!(&format!("{}_lanes_16", $id), $dim, $group, $data, $distance_fn, 16);
        bench_dual_dist_lanes_fn!(&format!("{}_lanes_32", $id), $dim, $group, $data, $distance_fn, 32);
        bench_dual_dist_lanes_fn!(&format!("{}_lanes_64", $id), $dim, $group, $data, $distance_fn, 64);
    }};
}

/// Helper macro to benchmark a distance function between two vectors.
macro_rules! bench_self_dist_lanes_fn {
    ($id:expr, $dim:expr, $group:expr, $data:expr, $distance_fn:ident, $lanes:expr) => {
        let car = $data.len();
        $group.throughput(criterion::Throughput::Elements(car as u64));

        let id = BenchmarkId::new($id, $dim);
        $group.bench_with_input(id, &$dim, |b, _| {
            b.iter_with_large_drop(|| {
                $data
                    .iter()
                    .map(|x| std::hint::black_box(distances_two::std_simd::$distance_fn::<_, _, $lanes>(x)))
                    .collect::<Vec<_>>()
            });
        });

        let car = car * car;
        $group.throughput(criterion::Throughput::Elements(car as u64));
    };
}

/// Helper macro to benchmark SIMD distance functions.
macro_rules! bench_simd_self_lanes_fns {
    ($feature:expr, $id:expr, $dim:expr, $group:expr, $data:expr, $distance_fn:ident) => {{
        // Lane-specific benchmarks.
        bench_self_dist_lanes_fn!(&format!("{}_lanes_4", $id), $dim, $group, $data, $distance_fn, 4);
        bench_self_dist_lanes_fn!(&format!("{}_lanes_8", $id), $dim, $group, $data, $distance_fn, 8);
        bench_self_dist_lanes_fn!(&format!("{}_lanes_16", $id), $dim, $group, $data, $distance_fn, 16);
        bench_self_dist_lanes_fn!(&format!("{}_lanes_32", $id), $dim, $group, $data, $distance_fn, 32);
        bench_self_dist_lanes_fn!(&format!("{}_lanes_64", $id), $dim, $group, $data, $distance_fn, 64);
    }};
}

/// Helper macro to benchmark SIMD distance functions.
macro_rules! bench_all_simd_lanes_fns {
    ($feature:expr, $dim:expr, $group:expr, $data:expr) => {{
        bench_simd_lanes_fns!($feature, "l2", $dim, $group, $data, euclidean);
        bench_simd_lanes_fns!($feature, "dot", $dim, $group, $data, dot_product);
        bench_simd_lanes_fns!($feature, "cosine", $dim, $group, $data, cosine);
        bench_simd_self_lanes_fns!($feature, "norm", $dim, $group, $data, norm_l2);
    }};
}

/// Helper macro to benchmark several distance functions.
macro_rules! bench_many_dist_fns {
    ($seed:expr, $car:expr, $max_dim_pow:expr, $type:ty, $c:expr) => {
        let group_name = concat!("vectors_", stringify!($type));
        let mut group = $c.benchmark_group(group_name);

        config_group(&mut group, $car);

        let dims = (0..=$max_dim_pow).map(|d| 1_000 * 2_u32.pow(d) as usize).collect::<Vec<_>>();

        for &dim in &dims {
            let data = gen_data::<$type>($car, dim, 1.0, 2.0, $seed);
            bench_dual_dist_fn!("l2_naive", dim, &mut group, data, distances_two::vectors::euclidean);
            bench_dual_dist_fn!("dot_naive", dim, &mut group, data, distances_two::vectors::dot_product);
            bench_dual_dist_fn!("cosine_naive", dim, &mut group, data, distances_two::vectors::cosine);
            bench_self_dist_fn!("norm_naive", dim, &mut group, data, distances_two::vectors::norm_l2);

            #[cfg(feature = "blas")]
            {
                bench_dual_dist_fn!("l2_blas", dim, &mut group, data, distances_two::blas::euclidean);
                bench_dual_dist_fn!("dot_blas", dim, &mut group, data, distances_two::blas::dot_product);
                bench_dual_dist_fn!("cosine_blas", dim, &mut group, data, distances_two::blas::cosine);
                bench_self_dist_fn!("norm_blas", dim, &mut group, data, distances_two::blas::norm_l2);
            }

            #[cfg(feature = "simd-128")]
            bench_simd_fns!("simd-128", dim, &mut group, data);
            #[cfg(feature = "simd-256")]
            bench_simd_fns!("simd-256", dim, &mut group, data);
            #[cfg(feature = "simd-512")]
            bench_simd_fns!("simd-512", dim, &mut group, data);
            #[cfg(feature = "simd-1024")]
            bench_simd_fns!("simd-1024", dim, &mut group, data);

            #[cfg(any(feature = "simd-128", feature = "simd-256", feature = "simd-512", feature = "simd-1024"))]
            bench_all_simd_lanes_fns!("std-simd", dim, &mut group, data);
        }

        group.finish();
    };
}

fn vector_distances(c: &mut Criterion) {
    let seed = 42_u64;
    let car = 100; // Number of vectors to compare. The number of distance computations is car^2.
    let max_dim_pow = 0_u32; // Max dimension is 1000 * 2^max_dim_pow.

    bench_many_dist_fns!(seed, car, max_dim_pow, f32, c);
    bench_many_dist_fns!(seed, car, max_dim_pow, f64, c);
}

criterion_group!(benches, vector_distances);
criterion_main!(benches);
