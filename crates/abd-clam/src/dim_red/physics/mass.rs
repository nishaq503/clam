//! Masses in the mass-spring system.

use distances::Number;
use rand::Rng;

use crate::Cluster;

/// A `Mass` in the mass-spring system for dimensionality reduction.
///
/// A `Mass` represents a `Cluster` in the reduced space, and is defined by its:
///
/// - `position`: The position of the `Mass` in the reduced space.
/// - `velocity`: The velocity of the `Mass` in the reduced space.
/// - `mass`: The mass of the `Mass`.
///
/// A `Mass` also stores state information for referencing back to the `Cluster`
/// it represents:
///
/// - `offset`: The offset of the `Cluster`.
/// - `cardinality`: The cardinality of the `Cluster`.
/// - `arg_center`: The index of the center of the `Cluster`.
///
/// The `Mass` also stores the force being applied to it, which is used to
/// update the position and velocity of the `Mass`.
///
/// # Type Parameters
///
/// - `DIM`: The dimensionality of the reduced space.
#[derive(Debug, Clone)]
pub struct Mass<const DIM: usize> {
    /// The offset of the `Cluster`.
    offset: usize,
    /// The cardinality of the `Cluster`.
    cardinality: usize,
    /// The index of the center of the `Cluster`.
    arg_center: usize,
    /// The position of the `Mass` in the reduced space.
    position: [f32; DIM],
    /// The velocity of the `Mass` in the reduced space.
    velocity: [f32; DIM],
    /// The force being applied to the `Mass`.
    force: [f32; DIM],
    /// The mass of the `Mass`.
    m: f32,
}

impl<const DIM: usize> Mass<DIM> {
    /// Constructs a `Mass` to represent a `Cluster`.
    ///
    /// This assigns the `position` and `velocity` of the `Mass` to be the zero
    /// vector, and the `mass` to be the cardinality of the `UniBall`.
    pub fn from_cluster<U: Number, C: Cluster<U>>(c: &C) -> Self {
        Self {
            offset: c.offset(),
            cardinality: c.cardinality(),
            arg_center: c.arg_center(),
            position: [0.0; DIM],
            velocity: [0.0; DIM],
            force: [0.0; DIM],
            m: c.cardinality().as_f32(),
        }
    }

    /// Creates a new `Mass`.
    #[must_use]
    pub fn new(offset: usize, cardinality: usize, arg_center: usize) -> Self {
        Self {
            offset,
            cardinality,
            arg_center,
            position: [0.0; DIM],
            velocity: [0.0; DIM],
            force: [0.0; DIM],
            m: cardinality.as_f32(),
        }
    }

    /// Returns the offset of the `Cluster`.
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
    }

    /// Returns the cardinality of the `Cluster`.
    #[must_use]
    pub const fn cardinality(&self) -> usize {
        self.cardinality
    }

    /// Returns the index of the center of the `Cluster`.
    #[must_use]
    pub const fn arg_center(&self) -> usize {
        self.arg_center
    }

    /// Gets the hash-key of the `Mass`.
    #[must_use]
    pub const fn hash_key(&self) -> (usize, usize) {
        (self.offset, self.cardinality)
    }

    /// Returns the position of the `Mass`.
    #[must_use]
    pub const fn position(&self) -> &[f32; DIM] {
        &self.position
    }

    /// Sets the position of the `Mass`.
    ///
    /// It is the user's responsibility to ensure that the `position` has no
    /// `NaN` or `Infinite` values.
    pub fn set_position(&mut self, position: [f32; DIM]) {
        self.position = position;
    }

    /// Resets the `velocity` of the `Mass` to the zero vector.
    pub fn reset_velocity(&mut self) {
        self.velocity = [0.0; DIM];
    }

    /// Returns the distance vector from this `Mass` to another `Mass`.
    pub fn distance_vector_to(&self, other: &Self) -> [f32; DIM] {
        let mut distance_vector = [0.0; DIM];
        for ((d, &p), &o) in distance_vector
            .iter_mut()
            .zip(self.position.iter())
            .zip(other.position.iter())
        {
            *d = o - p;
        }
        distance_vector
    }

    /// Returns the distance from this `Mass` to another `Mass`.
    pub fn current_distance_to(&self, other: &Self) -> f32 {
        self.distance_vector_to(other)
            .iter()
            .map(|x| x.powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Returns the unit vector from this `Mass` to another `Mass`.
    pub fn unit_vector_to(&self, other: &Self) -> [f32; DIM] {
        let mut vector = self.distance_vector_to(other);

        let magnitude = vector.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        if magnitude > (f32::EPSILON * DIM.as_f32()) {
            for d in &mut vector {
                *d /= magnitude;
            }
        } else {
            // Choose a random axis to move along
            vector = [0.0; DIM];
            let dim = rand::thread_rng().gen_range(0..DIM);
            let sign = if rand::thread_rng().gen_bool(0.5) { 1.0 } else { -1.0 };
            vector[dim] = sign;
        }
        vector
    }

    /// Adds the given force to the force being applied to the `Mass`.
    pub fn add_force(&mut self, force: [f32; DIM]) {
        // sf: self.force, f: force
        for (sf, &f) in self.force.iter_mut().zip(force.iter()) {
            *sf += f;
        }
    }

    /// Subtracts the given force from the force being applied to the `Mass`.
    pub fn sub_force(&mut self, force: [f32; DIM]) {
        // sf: self.force, f: force
        for (sf, &f) in self.force.iter_mut().zip(force.iter()) {
            *sf -= f;
        }
    }

    /// Applies the force being applied to the `Mass`, for one time-step.
    ///
    /// After applying the force, the position and velocity of the `Mass` are
    /// updated, and the force is reset to the zero vector.
    ///
    /// # Arguments
    ///
    /// - `dt`: The time-step to apply the force for.
    /// - `beta`: The damping factor.
    pub fn apply_force(&mut self, dt: f32, beta: f32) {
        for ((p, v), f) in self
            .position
            .iter_mut()
            .zip(self.velocity.iter_mut())
            .zip(self.force.iter_mut())
        {
            *f -= (*v) * beta;
            (*v) += ((*f) / self.m) * dt;
            (*p) += (*v) * dt;
            *f = 0.0;
        }
    }

    /// Get the kinetic energy of the `Mass`.
    pub fn kinetic_energy(&self) -> f32 {
        0.5 * self.m * self.velocity.iter().map(|v| v.powi(2)).sum::<f32>()
    }
}
