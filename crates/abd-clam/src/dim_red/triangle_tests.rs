use crate::{Dataset, Instance};
use distances::Number;
use mt_logger::{mt_log, Level};
use rand::prelude::SliceRandom;

use super::physics::Mass;

/// Type alias representing a list of three edges
pub type Triangle = [Edge; 3];
/// An Edge is a tuple storing its id and its length
pub type Edge = (usize, f32);

/// Runs through a vec of tests and returns a floating point value for each
pub fn triangle_accuracy_tests<U: Number, I: Instance, D: Dataset<I, U>, const DIM: usize>(
    masses: &super::physics::MassMap<DIM>,
    data: &D,
    test_callbacks: [fn(Triangle, Triangle) -> f32; 2],
) -> Option<[f32; 2]> {
    if masses.len() < 3 {
        return None;
    }
    let mut masses_copy: Vec<_> = masses.iter().map(|m| m.1).collect();
    let mut valid_triangle_count: usize = 0;
    let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
    let mut metric_sums = [0.0f32; 2];
    for mass in masses.iter().map(|m| m.1) {
        masses_copy.partial_shuffle(&mut rng, 4);
        if let Some(chosen_masses) = choose_two_unique_with(&masses_copy, mass) {
            if let Some((ref_triangle, test_triangle)) = masses_to_triangles(&chosen_masses, data) {
                for (i, test_cb) in test_callbacks.iter().enumerate() {
                    metric_sums[i] += test_cb(ref_triangle, test_triangle);
                }
                valid_triangle_count += 1;
            } else {
                // try again once if invalid triangle
                masses_copy.partial_shuffle(&mut rng, 4);
                if let Some(chosen_masses) = choose_two_unique_with(&masses_copy, mass) {
                    if let Some((ref_triangle, test_triangle)) = masses_to_triangles(&chosen_masses, data) {
                        for (i, test_cb) in test_callbacks.iter().enumerate() {
                            metric_sums[i] += test_cb(ref_triangle, test_triangle);
                        }
                        valid_triangle_count += 1;
                    }
                }
            }
        }
    }

    if valid_triangle_count != masses.len() {
        let potential_triangle_count = masses.len();
        mt_log!(
            Level::Error,
            "valid triangle ratio = {valid_triangle_count} / {potential_triangle_count}"
        );
    }
    if valid_triangle_count == 0 || valid_triangle_count.as_f64() / masses.len().as_f64() < 0.5 {
        mt_log!(Level::Error, "valid triangle ratio critical");
        metric_sums.iter_mut().for_each(|val| *val = -1000.);
    } else {
        metric_sums
            .iter_mut()
            .for_each(|val| *val /= valid_triangle_count.as_f32());
    }

    Some(metric_sums)
}

/// Helper function returns three unique `Mass` variables
#[must_use]
pub fn choose_two_unique_with<'a, const DIM: usize>(
    masses: &'a [&'a Mass<DIM>],
    cur_mass: &'a Mass<DIM>,
) -> Option<[&'a Mass<DIM>; 3]> {
    if masses.len() < 3 {
        return None;
    }
    let mut triangle: [&Mass<DIM>; 3] = [cur_mass, masses[0], masses[1]];

    for m in triangle.iter_mut().skip(1) {
        if *m == cur_mass {
            *m = masses[2];
        }
    }

    Some(triangle)
}

/// helper function converts a vec of `Mass` to a tuple of Vecs where one stores the metric edge lengths and the other stores the physical edge lengths
pub fn masses_to_triangles<U: Number, I: Instance, D: Dataset<I, U>, const DIM: usize>(
    masses: &[&Mass<DIM>; 3],
    data: &D,
) -> Option<(Triangle, Triangle)> {
    let ref_triangle: [(usize, f32); 3] = [
        (0, masses[0].distance_to_other(data, masses[1]).as_f32()),
        (1, masses[1].distance_to_other(data, masses[2]).as_f32()),
        (2, masses[0].distance_to_other(data, masses[2]).as_f32()),
    ];

    let test_triangle: [(usize, f32); 3] = [
        (0, masses[0].current_distance_to(masses[1])),
        (1, masses[1].current_distance_to(masses[2])),
        (2, masses[0].current_distance_to(masses[2])),
    ];

    if is_valid_triangle(&ref_triangle) && is_valid_triangle(&test_triangle) {
        return Some((ref_triangle, test_triangle));
    }

    None
}

/// Accepts two lists of edges and returns if correspnding edges are in the correct order when sorted by length
pub fn are_triangles_equivalent(mut ref_edges: [(usize, f32); 3], mut test_edges: [(usize, f32); 3]) -> f32 {
    ref_edges.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    test_edges.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut correct_edge_count = 0;
    for (e1, e2) in ref_edges.iter().zip(test_edges.iter()) {
        if e1.0 == e2.0 {
            correct_edge_count += 1;
        }
    }

    if correct_edge_count == 3 {
        1.0
    } else {
        0.0
    }
}

/// Function that computes the average edge distortion of triangles in the graph
pub fn calc_edge_distortion(ref_edges: [(usize, f32); 3], test_edges: [(usize, f32); 3]) -> f32 {
    let perimeter_ref: f32 = ref_edges.iter().map(|&(_, value)| value).sum();
    let perimeter_test: f32 = test_edges.iter().map(|&(_, value)| value).sum();

    let ref_percentages: Vec<f32> = ref_edges.iter().map(|&(_, val)| val / perimeter_ref).collect();

    let test_percentages: Vec<f32> = test_edges.iter().map(|&(_, val)| val / perimeter_test).collect();

    let distortion: f32 = ref_percentages
        .iter()
        .zip(test_percentages.iter())
        .map(|(&x, &y)| (y - x).abs())
        .sum();

    distortion / 3.0
}

/// Function that computes the average distortion of triangle angles in the graph
#[allow(dead_code)]
pub fn calc_angle_distortion(ref_edges: [(usize, f32); 3], test_edges: [(usize, f32); 3]) -> f32 {
    let test_angles: [f32; 3] = compute_angles_from_edge_lengths(&test_edges);

    let ref_angles: [f32; 3] = compute_angles_from_edge_lengths(&ref_edges);
    let ref_angle_sum: f32 = {
        let sum = ref_angles.iter().sum();

        // if sum >= 181.0 {
        //     mt_log!(Level::Error, "Angle sum greater than 181");
        //     // panic!();
        // }
        if sum > 180.05 {
            mt_log!(Level::Error, "ref angle sum: {sum}");
            180.0
        } else {
            sum
        }
    };

    let test_angle_sum: f32 = {
        let sum = test_angles.iter().sum();
        // if sum >= 181.0 {
        //     mt_log!(Level::Error, "Angle sum greater than 181");
        //     panic!();
        // }
        if sum > 180.005 {
            mt_log!(Level::Error, "test angle sum: {sum}");
            180.0
        } else {
            sum
        }
    };
    // let mut err = false;
    // if ref_angle_sum > 180. + 0.008 {
    //     err = true;
    //     mt_log!(Level::Error, "ref angle sum: {ref_angle_sum}");
    // }

    // if test_angle_sum > 180. + 0.008 {
    //     err = true;

    //     mt_log!(Level::Error, "ref angle sum: {test_angle_sum}");
    // }

    let ref_percentages: Vec<f32> = ref_angles.iter().map(|&val| val / ref_angle_sum).collect();

    let test_percentages: Vec<f32> = test_angles.iter().map(|&val| val / test_angle_sum).collect();

    // if err {
    //     mt_log!(Level::Error, "ref perc: {ref_percentages:?}");
    //     mt_log!(Level::Error, "test perc: {test_percentages:?}");
    // }

    let distortion: f32 = ref_percentages
        .iter()
        .zip(test_percentages.iter())
        .map(|(&x, &y)| (y - x).abs())
        .sum();
    // if err {
    //     mt_log!(Level::Error, "disortion: {distortion}");
    // }
    distortion / 3.0
}

/// Helper functions computes angles of a triangle given only its edge lengths
pub fn compute_angles_from_edge_lengths(edges: &Triangle) -> [f32; 3] {
    // Extract edge lengths for better readability
    let a = edges[0].1;
    let b = edges[1].1;
    let c = edges[2].1;

    // Compute squares of edge lengths for easier calculations
    let a_squared = a * a;
    let b_squared = b * b;
    let c_squared = c * c;

    // Calculate cosines of angles using the law of cosines
    let cosine_a = ((b_squared + c_squared - a_squared) / (2. * b * c)).clamp(-1., 1.);
    let cosine_b = ((a_squared + c_squared - b_squared) / (2. * a * c)).clamp(-1., 1.);
    let cosine_c = ((a_squared + b_squared - c_squared) / (2. * a * b)).clamp(-1., 1.);

    assert!(!cosine_a.is_nan());
    assert!(!cosine_b.is_nan());
    assert!(!cosine_c.is_nan());

    let angle_a = cosine_a.acos();
    let angle_b = cosine_b.acos();
    let angle_c = cosine_c.acos();
    if angle_a.is_nan() {
        mt_log!(Level::Error, "cosa: {cosine_a}");
    }
    assert!(!angle_a.is_nan());
    if angle_b.is_nan() {
        mt_log!(Level::Error, "cosb: {cosine_b}");
    }
    assert!(!angle_b.is_nan());
    if angle_c.is_nan() {
        mt_log!(Level::Error, "cosc: {cosine_c}");
    }
    assert!(!angle_c.is_nan());

    let angle_a = angle_a.to_degrees();
    let angle_b = angle_b.to_degrees();
    let angle_c = angle_c.to_degrees();

    assert!(!angle_a.is_nan());
    assert!(!angle_b.is_nan());
    assert!(!angle_c.is_nan());

    // if angle_a + angle_b + angle_c > 180.0 {
    //     panic!(
    //         "angle sums greater than 180 {}",
    //         angle_a + angle_b + angle_c
    //     )
    // }

    [angle_a, angle_b, angle_c]
}

/// Tests that triangle holds for the inequality theorem
pub fn is_valid_triangle(edges: &[(usize, f32)]) -> bool {
    if edges.len() != 3 {
        return false;
    }

    for e in edges {
        if e.1 == 0. {
            mt_log!(Level::Error, "edge len is 0");
            return false;
        }
    }

    let (a, b, c) = (edges[0].1, edges[1].1, edges[2].1);

    // Triangle inequality theorem: the sum of the lengths of any two sides of a triangle must be greater than the length of the third side.
    (a + b > c) && (a + c > b) && (b + c > a)
}

// pub fn print_triangle_inequality(triangle : &Triangle){
//     let (a, b, c) = (triangle[0].1, triangle[1].1, triangle[2].1);
//     mt_log!(Level::Info, "(a,b,c) {a} + {b} > {c}, (a,c,b){a} + {c} > {b}, (c,b,a) {c} + {b} > {a}")
// }
