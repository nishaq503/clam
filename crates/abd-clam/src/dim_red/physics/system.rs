//! The mass-spring system.

use std::collections::HashSet;

use distances::Number;
use mt_logger::{mt_log, Level};
use rand::prelude::*;
use rayon::prelude::*;

use crate::{chaoda::Graph, Dataset, Instance, VecDataset};

use super::{Mass, Spring};

/// The `System` of `Mass`es and `Spring`s.
pub struct System<U: Number, const DIM: usize> {
    /// A sorted collection of `Mass`es.
    masses: Vec<Mass<DIM>>,
    /// A collection of `Spring`s.
    springs: HashSet<Spring<U, DIM>>,
    /// The damping factor.
    beta: f32,
    /// The logs of the `System` for each time-step. These store the kinetic
    /// energy, potential energy, and total energy.
    logs: Vec<[f32; 3]>,
}

impl<U: Number, const DIM: usize> System<U, DIM> {
    /// Create a new `Mass`-`Spring` `System` from a CHAODA graph.
    ///
    /// # Arguments
    ///
    /// - `g`: The CHAODA graph.
    /// - `k`: The spring constant of all `Spring`s.
    /// - `beta`: The damping factor, applied to the velocity of the `Mass`es.
    #[must_use]
    pub fn from_graph(g: &Graph<U>, k: f32, beta: f32, seed: Option<u64>) -> Self {
        // Create the masses
        let masses = g
            .iter_clusters()
            .map(|&(offset, cardinality, arg_center)| Mass::new(offset, cardinality, arg_center))
            .collect();

        // Create the springs
        let springs = g
            .iter_neighbors()
            .enumerate()
            .flat_map(|(i, neighbors)| neighbors.iter().map(move |&(j, l0)| Spring::new(i, j, k, l0)))
            .collect();

        let logs = Vec::new();

        // Create the system
        let system = Self {
            masses,
            springs,
            beta,
            logs,
        };

        // Get the maximum l0 to set the initial positions of the masses
        let cube_side_len = system
            .springs
            .iter()
            .map(Spring::l0_f32)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
            .unwrap_or(1.0);

        // Initialize the system with random positions
        system.init_random(cube_side_len, seed)
    }

    /// Set random positions for the `Mass`es.
    ///
    /// The positions are set randomly within a cube centered at the origin with
    /// side length `cube_side_len`.
    ///
    /// # Arguments
    ///
    /// - `cube_side_len`: The side length of the cube.
    #[must_use]
    pub fn init_random(mut self, cube_side_len: f32, seed: Option<u64>) -> Self {
        let mut rng = seed.map_or_else(StdRng::from_entropy, StdRng::seed_from_u64);

        let (min_, max_) = (-cube_side_len / 2.0, cube_side_len / 2.0);
        for m in &mut self.masses {
            let mut position = [0.0; DIM];
            for p in &mut position {
                *p = rng.gen_range(min_..max_);
            }
            m.set_position(position);
        }

        self.update_springs()
    }

    /// Update the lengths of the `Spring`s and the forces exerted by them.
    #[must_use]
    pub fn update_springs(mut self) -> Self {
        self.springs = self
            .springs
            .into_par_iter()
            .map(|mut s| {
                s.update_length(&self.masses);
                s
            })
            .collect();

        self
    }

    /// Add a `Spring` between two `Mass`es.
    ///
    /// # Arguments
    ///
    /// - `i`: The index of the first `Mass`.
    /// - `j`: The index of the second `Mass`.
    /// - `k`: The spring constant of the `Spring`.
    /// - `data`: The `Dataset` containing the `Instance`s.
    ///
    /// # Returns
    ///
    /// * `true` if the `Spring` was added.
    /// * `false` if the `Spring` was already present, in which case the spring
    ///  constant is updated.
    pub fn add_spring<I: Instance, D: Dataset<I, U>>(&mut self, i: usize, j: usize, k: f32, data: &D) -> bool {
        let l0 = data.one_to_one(self.masses[i].arg_center(), self.masses[j].arg_center());
        let mut s = Spring::new(i, j, k, l0);

        let was_replaced = !self.springs.remove(&s);
        s.update_length(&self.masses);
        self.springs.insert(s);

        was_replaced
    }

    /// Get the `Mass`es in the `System`.
    #[must_use]
    pub fn masses(&self) -> &[Mass<DIM>] {
        &self.masses
    }

    /// Get the `Mass`es in the `System` as mutable.
    #[must_use]
    pub fn masses_mut(&mut self) -> &mut [Mass<DIM>] {
        &mut self.masses
    }

    /// Get the pairs of `Mass`es that are connected by `Spring`s.
    #[must_use]
    pub fn mass_pairs(&self) -> Vec<[&Mass<DIM>; 2]> {
        self.springs
            .iter()
            .map(|s| {
                let [i, j] = s.get_arg_masses();
                [&self.masses[i], &self.masses[j]]
            })
            .collect()
    }

    /// For a given mass (by its index), get the indices of the masses it is
    /// connected to and the `Spring`s connecting them.
    #[must_use]
    pub fn connected_masses(&self, i: usize) -> Vec<(usize, &Spring<U, DIM>)> {
        self.springs
            .par_iter()
            .filter_map(|s| {
                let [m1, m2] = s.get_arg_masses();
                if i == m1 {
                    Some((m2, s))
                } else if i == m2 {
                    Some((m1, s))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Update the `System` by one time-step.
    ///
    /// This will calculate the forces exerted by the `Spring`s and apply them
    /// to the `Mass`es. The `Mass`es will then be moved according to the
    /// forces. Finally, the `Spring`s will be updated to reflect the new
    /// positions of the `Mass`es.
    ///
    /// # Arguments
    ///
    /// - `dt`: The time-step.
    #[must_use]
    pub fn update_step(mut self, dt: f32) -> Self {
        // Calculate the forces exerted by the springs
        let forces: Vec<(usize, usize, [f32; DIM])> = self
            .springs
            .par_iter()
            .map(|s| {
                let [i, j] = s.get_arg_masses();
                let unit_vector = self.masses[i].unit_vector_to(&self.masses[j]);
                let f_mag: f32 = s.f();
                let mut force = [0.0; DIM];
                for (f, &uv) in force.iter_mut().zip(unit_vector.iter()) {
                    *f = f_mag * uv;
                }
                (i, j, force)
            })
            .collect();

        // Accumulate the forces for each mass
        for (i, j, force) in forces {
            self.masses[i].add_force(force);
            self.masses[j].sub_force(force);
        }

        // Apply the forces to the masses
        self.masses.par_iter_mut().for_each(|m| m.apply_force(dt, self.beta));

        // Update the springs
        // TODO: Merge this iteration with the first one
        self.springs = self
            .springs
            .into_par_iter()
            .map(|mut s| {
                s.update_length(&self.masses);
                s
            })
            .collect();

        self.update_logs();

        self
    }

    /// Simulate the `System` for a given number of time-steps.
    ///
    /// # Arguments
    ///
    /// - `dt`: The time-step.
    /// - `steps`: The number of time-steps to simulate.
    #[must_use]
    pub fn evolve(mut self, dt: f32, steps: usize) -> Self {
        self.update_logs();

        for _ in 0..steps {
            self = self.update_step(dt);
        }
        self
    }

    /// Simulate the `System` for a given number of time-steps, and save the
    /// intermediate states and final logs.
    ///
    /// # Arguments
    ///
    /// - `dt`: The time-step.
    /// - `steps`: The number of time-steps to simulate.
    /// - `save_every`: The number of time-steps between each saved state.
    /// - `data`: The `Dataset` containing the `Instance`s.
    /// - `path`: The path to the directory where the intermediate states will
    /// be saved.
    /// - `name`: The name of the `VecDataset` containing the intermediate states.
    ///
    /// # Returns
    ///
    /// The `System` in its final state.
    ///
    /// # Errors
    ///
    /// * If there is an error saving the intermediate states.
    pub fn evolve_with_saves<I: Instance, D: Dataset<I, U>>(
        mut self,
        dt: f32,
        steps: usize,
        save_every: usize,
        data: &D,
        path: &std::path::Path,
        name: &str,
    ) -> Result<Self, String> {
        self.update_logs();

        for i in 0..steps {
            if i % save_every == 0 {
                mt_log!(Level::Debug, "{name}: Saving step {i}/{steps}");

                self.get_reduced_embedding(data, name)
                    .to_npy(&path.join(format!("{i}.npy")))?;
            }
            self = self.update_step(dt);
        }

        Ok(self)
    }

    /// Update the `logs`
    fn update_logs(&mut self) {
        let kinetic_energy = self.kinetic_energy();
        let potential_energy = self.potential_energy();
        let total_energy = kinetic_energy + potential_energy;
        self.logs.push([kinetic_energy, potential_energy, total_energy]);
    }

    /// Get the logs of the `System` for each time-step.
    #[must_use]
    pub fn logs(&self) -> &[[f32; 3]] {
        &self.logs
    }

    /// Clear the logs of the `System`.
    pub fn clear_logs(&mut self) {
        self.logs.clear();
    }

    /// Get the dimension-reduced embedding from this `System`.
    ///
    /// This will return a `VecDataset` with the following properties:
    ///
    /// - The instances are the positions of the `Mass`es.
    /// - The metadata are the indices of the centers of the `Cluster`s
    ///  represented by the `Mass`es.
    /// - The distance function is the Euclidean distance.
    ///
    /// # Arguments
    ///
    /// - `data`: The original `Dataset` containing the `Instance`s.
    /// - `name`: The name of the new `VecDataset`.
    #[must_use]
    pub fn get_reduced_embedding<I: Instance, D: Dataset<I, U>>(
        &self,
        data: &D,
        name: &str,
    ) -> VecDataset<Vec<f32>, f32, usize> {
        let masses = {
            let mut masses = self.masses.clone();
            masses.sort_by_key(Mass::offset);
            masses
        };

        let positions = {
            let positions = masses
                .par_iter()
                .flat_map(|m| (0..m.cardinality()).map(move |_| m.position()).collect::<Vec<_>>())
                .collect::<Vec<_>>();

            let permutation = data
                .permuted_indices()
                .map_or_else(|| (0..data.cardinality()).collect(), <[usize]>::to_vec);
            let mut positions = permutation.into_iter().zip(positions).collect::<Vec<_>>();
            positions.sort_by(|(a, _), (b, _)| a.cmp(b));

            positions.into_iter().map(|(_, p)| p.to_vec()).collect::<Vec<_>>()
        };

        VecDataset::new(
            name.to_string(),
            positions,
            |x: &Vec<f32>, y: &Vec<f32>| distances::vectors::euclidean(x, y),
            false,
        )
    }

    /// Get the total potential energy of the `System`.
    #[must_use]
    pub fn potential_energy(&self) -> f32 {
        self.springs.par_iter().map(Spring::potential_energy).sum()
    }

    /// Get the total kinetic energy of the `System`.
    #[must_use]
    pub fn kinetic_energy(&self) -> f32 {
        self.masses.par_iter().map(Mass::kinetic_energy).sum()
    }

    /// Get the total energy of the `System`.
    #[must_use]
    pub fn total_energy(&self) -> f32 {
        self.potential_energy() + self.kinetic_energy()
    }
}
