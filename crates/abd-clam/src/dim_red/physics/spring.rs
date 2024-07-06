//! The spring for the mass-spring system.

use std::collections::HashMap;

use distances::Number;

use super::Mass;

/// A spring in the mass-spring system.
///
/// The spring is defined by its:
///
/// - spring constant `k`, i.e. the stiffness of the `Spring`,
/// - rest length `l0`, i.e. the distance between the two connected `Cluster`s in the original embedding space.
/// - current length `l`, i.e. the distance between the two connected `Mass`es in the reduced space.
///
/// # Type Parameters
///
/// - `U`: The type of distance values in the original embedding space.
/// - `DIM`: The dimensionality of the reduced space.
#[derive(Debug, Clone)]
pub struct Spring<U: Number, const DIM: usize> {
    /// The hash-key of the first `Mass` connected by the `Spring` in the `System`.
    i: (usize, usize),
    /// The hash-key of the second `Mass` connected by the `Spring` in the `System`.
    j: (usize, usize),
    /// The spring constant of the `Spring`.
    k: f32,
    /// The length of the `Spring` in the original embedding space.
    l0: U,
    /// The length of the `Spring` in the original embedding space cast to `f32`.
    l0_f32: f32,
    /// The length of the `Spring` in the reduced space.
    l: f32,
    /// The magnitude force exerted by the `Spring`.
    f: f32,
}

impl<U: Number, const DIM: usize> Spring<U, DIM> {
    /// Create a new `Spring`.
    #[allow(clippy::many_single_char_names)]
    pub fn new(i: (usize, usize), j: (usize, usize), k: f32, l0: U) -> Self {
        // Order the masses
        let (i, j) = if i < j { (i, j) } else { (j, i) };

        let mut s = Self {
            i,
            j,
            k,
            l0,
            l0_f32: l0.as_f32(),
            l: l0.as_f32(),
            f: 0.0,
        };
        s.update_force();
        s
    }

    /// Get the rest length of the `Spring`.
    pub const fn l0(&self) -> U {
        self.l0
    }

    /// Get the rest length of the `Spring` as a `f32`.
    pub const fn l0_f32(&self) -> f32 {
        self.l0_f32
    }

    /// Get the spring constant of the `Spring`.
    pub const fn k(&self) -> f32 {
        self.k
    }

    /// Get the magnitude of the force exerted by the `Spring`.
    pub const fn f(&self) -> f32 {
        self.f
    }

    /// Get the displacement of the `Spring`, i.e. the rest length minus the current length.
    pub fn dx(&self) -> f32 {
        self.l0_f32 - self.l
    }

    /// Get the indices of the masses connected by the `Spring`.
    pub const fn hash_key(&self) -> ((usize, usize), (usize, usize)) {
        (self.i, self.j)
    }

    /// Set the indices of the masses connected by the `Spring`.
    pub fn set_arg_masses(&mut self, m1: (usize, usize), m2: (usize, usize)) {
        if m1.0 < m2.0 {
            self.i = m1;
            self.j = m2;
        } else if m1.0 > m2.0 {
            self.i = m2;
            self.j = m1;
        } else if m1.1 < m2.1 {
            self.i = m1;
            self.j = m2;
        } else {
            self.i = m2;
            self.j = m1;
        }
    }

    /// Set the spring constant of the `Spring`.
    pub fn set_spring_constant(&mut self, k: f32) {
        self.k = k;
    }

    /// Set the current length of the `Spring`.
    ///
    /// This will also update the magnitude of the force exerted by the `Spring`.
    pub fn update_length(&mut self, masses: &HashMap<(usize, usize), Mass<DIM>>) {
        self.l = masses[&self.i].current_distance_to(&masses[&self.j]);
        self.update_force();
    }

    /// Recalculate the magnitude of the force exerted by the `Spring`.
    fn update_force(&mut self) {
        self.f = -self.k * self.dx();
    }

    /// Get the potential energy of the `Spring`.
    pub fn potential_energy(&self) -> f32 {
        0.5 * self.k * self.dx().powi(2)
    }
}

impl<U: Number, const DIM: usize> core::hash::Hash for Spring<U, DIM> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.hash_key().hash(state);
    }
}

impl<U: Number, const DIM: usize> PartialEq for Spring<U, DIM> {
    fn eq(&self, other: &Self) -> bool {
        self.hash_key() == other.hash_key()
    }
}

impl<U: Number, const DIM: usize> Eq for Spring<U, DIM> {}
