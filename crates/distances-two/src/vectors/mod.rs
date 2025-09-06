//! Distance functions for vectors of the same dimensionality.

mod angular;
mod minkowski;

pub use angular::{cosine, cosine_normalized, cosine_similarity, cosine_similarity_normalized, dot_product};
pub use minkowski::{chebyshev, euclidean, euclidean_sq, manhattan, minkowski, norm_l2, norm_l2_sq};
