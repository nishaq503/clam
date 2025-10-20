//! Benchmarks for vector distances.

use std::hint::black_box;

use criterion::*;
use rand::prelude::*;

use distances_two::simd;

fn config_group(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>, car: usize) {
    let n_elements = car * car;
    group.sample_size(100);
    group.throughput(criterion::Throughput::Elements(n_elements as u64));

    let plot_config = criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic);
    group.plot_config(plot_config);
}

fn simd_1024_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("VectorsF32");

    let seed = 42;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let (car, min_val, max_val) = (100, -10_f32, 10_f32);
    config_group(&mut group, car);

    for d in 0..=3 {
        let dim = 1_000 * 2_u32.pow(d) as usize;

        let data: Vec<Vec<f32>> = (0..car)
            .map(|_| {
                (0..dim)
                    .map(|_| rand::Rng::random_range(&mut rng, min_val..=max_val))
                    .collect()
            })
            .collect();

        let id = BenchmarkId::new("Simd1024L2Sq", dim);
        group.bench_with_input(id, &dim, |b, _| {
            b.iter_with_large_drop(|| {
                data.iter()
                    .map(|x| {
                        data.iter()
                            .map(|y| black_box(simd::euclidean_sq(x, y)))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            });
        });

        let id = BenchmarkId::new("Simd1024L2", dim);
        group.bench_with_input(id, &dim, |b, _| {
            b.iter_with_large_drop(|| {
                data.iter()
                    .map(|x| {
                        data.iter()
                            .map(|y| black_box(simd::euclidean(x, y)))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            });
        });
    }

    group.finish();
}

fn simd_1024_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("VectorsF64");

    let seed = 42;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let (car, min_val, max_val) = (100, -10_f64, 10_f64);
    config_group(&mut group, car);

    for d in 0..=3 {
        let dim = 1_000 * 2_u32.pow(d) as usize;

        let data: Vec<Vec<f64>> = (0..car)
            .map(|_| {
                (0..dim)
                    .map(|_| rand::Rng::random_range(&mut rng, min_val..=max_val))
                    .collect()
            })
            .collect();

        let id = BenchmarkId::new("Simd1024L2Sq", dim);
        group.bench_with_input(id, &dim, |b, _| {
            b.iter_with_large_drop(|| {
                data.iter()
                    .map(|x| {
                        data.iter()
                            .map(|y| black_box(simd::euclidean_sq(x, y)))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            });
        });

        let id = BenchmarkId::new("Simd1024L2", dim);
        group.bench_with_input(id, &dim, |b, _| {
            b.iter_with_large_drop(|| {
                data.iter()
                    .map(|x| {
                        data.iter()
                            .map(|y| black_box(simd::euclidean(x, y)))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            });
        });
    }

    group.finish();
}

criterion_group!(benches, simd_1024_f32, simd_1024_f64);
criterion_main!(benches);
