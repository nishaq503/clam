//! Tests for basic vector distance functions.

use float_eq::assert_float_eq;
use test_case::test_case;

#[macro_use]
mod naive_impls;

#[test_case(10, 2; "10x2")]
#[test_case(100, 10; "100x10")]
#[test_case(100, 100; "100x100")]
#[test_case(100, 1000; "100x1000")]
fn vectors(car: usize, dim: usize) {
    let seed = 42;

    let data = naive_impls::gen_data::<f32>(car, dim, -1.0, 1.0, seed);
    for x in &data {
        for y in &data {
            check_distances!(x, y, 1e-5_f32);
        }
    }

    let data = naive_impls::gen_data::<f64>(car, dim, -1.0, 1.0, seed);
    for x in &data {
        for y in &data {
            check_distances!(x, y, 1e-5_f64);
        }
    }
}
