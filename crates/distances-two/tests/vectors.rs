//! Tests for basic vector distance functions.

use float_eq::assert_float_eq;
use rand::prelude::*;
use test_case::test_case;

mod naive_impls;

#[test_case(10, 2; "10x2")]
#[test_case(100, 10; "100x10")]
#[test_case(100, 100; "100x100")]
#[test_case(100, 1000; "100x1000")]
fn vectors_f32(car: usize, dim: usize) {
    let seed = 42;
    let (min_val, max_val) = (-1_f32, 1_f32);

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let data: Vec<Vec<f32>> = (0..car)
        .map(|_| {
            (0..dim)
                .map(|_| rand::Rng::random_range(&mut rng, min_val..=max_val))
                .collect()
        })
        .collect();

    for x in &data {
        for y in &data {
            let e_l1 = naive_impls::l1(x, y);
            let a_l1: f32 = distances_two::vectors::manhattan(x, y);
            let ratio = if e_l1 != 0.0 {
                (e_l1 - a_l1).abs() / e_l1.abs()
            } else {
                0.0
            };
            assert_float_eq!(ratio, 0.0, abs <= 1e-5);

            let a_l1_minkowski = distances_two::vectors::minkowski(x, y, 1.0);
            let ratio = if e_l1 != 0.0 {
                (e_l1 - a_l1_minkowski).abs() / e_l1.abs()
            } else {
                0.0
            };
            assert_float_eq!(ratio, 0.0, abs <= 1e-5);

            let e_l2_sq = naive_impls::l2_sq(x, y);
            let a_l2_sq: f32 = distances_two::vectors::euclidean_sq(x, y);
            let ratio = if e_l2_sq != 0.0 {
                (e_l2_sq - a_l2_sq).abs() / e_l2_sq.abs()
            } else {
                0.0
            };
            assert_float_eq!(ratio, 0.0, abs <= 1e-5);

            let e_l2 = naive_impls::l2(x, y);
            let a_l2: f32 = distances_two::vectors::euclidean(x, y);
            let ratio = if e_l2 != 0.0 {
                (e_l2 - a_l2).abs() / e_l2.abs()
            } else {
                0.0
            };
            assert_float_eq!(ratio, 0.0, abs <= 1e-5);

            let a_l2_minkowski = distances_two::vectors::minkowski(x, y, 2.0);
            let ratio = if e_l2 != 0.0 {
                (e_l2 - a_l2_minkowski).abs() / e_l2.abs()
            } else {
                0.0
            };
            assert_float_eq!(ratio, 0.0, abs <= 1e-5);

            let e_l3 = naive_impls::l3(x, y);
            let a_l3: f32 = distances_two::vectors::minkowski(x, y, 3.0);
            let ratio = if e_l3 != 0.0 {
                (e_l3 - a_l3).abs() / e_l3.abs()
            } else {
                0.0
            };
            assert_float_eq!(ratio, 0.0, abs <= 1e-5);

            let e_l4 = naive_impls::l4(x, y);
            let a_l4: f32 = distances_two::vectors::minkowski(x, y, 4.0);
            let ratio = if e_l4 != 0.0 {
                (e_l4 - a_l4).abs() / e_l4.abs()
            } else {
                0.0
            };
            assert_float_eq!(ratio, 0.0, abs <= 1e-5);

            let e_linf = naive_impls::l_inf(x, y);
            let a_linf: f32 = distances_two::vectors::chebyshev(x, y);
            let ratio = if e_linf != 0.0 {
                (e_linf - a_linf).abs() / e_linf.abs()
            } else {
                0.0
            };
            assert_float_eq!(ratio, 0.0, abs <= 1e-5);
        }
    }
}
