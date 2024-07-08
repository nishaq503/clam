//! The mass-spring system.

use distances::Number;
use mt_logger::{mt_log, Level};
use rand::prelude::*;
use rayon::prelude::*;

use crate::{
    chaoda::Graph,
    dim_red::triangle_tests::{self, triangle_accuracy_tests},
    Cluster, Dataset, Instance, VecDataset,
};

use super::{Mass, MassMap, Spring, SpringMap};

/// Represents the length of the log data - currently stores kinetic energy,
///  total energy, potential energy, edge equivalence, edge distortion, - angle distortion not included currently
pub type LogData = [f32; 5];

/// The `System` of `Mass`es and `Spring`s.
pub struct System<U: Number, const DIM: usize> {
    /// A sorted collection of `Mass`es.
    masses: MassMap<DIM>,
    /// A collection of `Spring`s.
    springs: SpringMap<U, DIM>,
    /// The damping factor.
    beta: f32,
    /// The logs of the `System` for each time-step. These store the kinetic
    /// energy, potential energy, and total energy.
    logs: Vec<LogData>,
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
        // Extract the clusters and their edges
        let clusters = g.iter_clusters().collect::<Vec<_>>();

        // Create the masses
        let masses = g
            .iter_clusters()
            .map(|&(offset, cardinality, arg_center)| Mass::new(offset, cardinality, arg_center))
            .map(|m| (m.hash_key(), m))
            .collect();

        // Create the springs
        let springs = g
            .iter_neighbors()
            .enumerate()
            .map(|(i, neighbors)| {
                let &(o, c, _) = clusters[i];
                ((o, c), neighbors)
            })
            .flat_map(|(i, neighbors)| {
                neighbors
                    .iter()
                    .map(|&(j, l0)| {
                        let &(o, c, _) = clusters[j];
                        ((o, c), l0)
                    })
                    .map(move |(j, l0)| Spring::new(i, j, k, l0))
            })
            .map(|s| (s.hash_key(), s))
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
            .values()
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
        for m in self.masses.values_mut() {
            let mut position = [0.0; DIM];
            for p in &mut position {
                *p = rng.gen_range(min_..max_);
            }
            m.set_position(position);
        }

        self.update_springs()
    }

    /// Get the `Mass`es in the `System`.
    #[must_use]
    pub const fn masses(&self) -> &MassMap<DIM> {
        &self.masses
    }

    /// Get the `Mass`es in the `System` as mutable.
    #[must_use]
    pub fn masses_mut(&mut self) -> &mut MassMap<DIM> {
        &mut self.masses
    }

    /// Get a single `Mass` by its offset and cardinality.
    #[must_use]
    pub fn get_mass(&self, offset: usize, cardinality: usize) -> Option<&Mass<DIM>> {
        self.masses.get(&(offset, cardinality))
    }

    /// Get a single `Mass` by its offset and cardinality as mutable.
    #[must_use]
    pub fn get_mass_mut(&mut self, offset: usize, cardinality: usize) -> Option<&mut Mass<DIM>> {
        self.masses.get_mut(&(offset, cardinality))
    }

    /// Add a `Mass` to the `System`. This does not add any `Spring`s.
    ///
    /// # Arguments
    ///
    /// - `c`: The `Cluster` to add as a `Mass`.
    /// - `data`: The `Dataset` containing the `Instance`s.
    /// - `position`: The initial position of the `Mass`.
    ///
    /// # Returns
    ///
    /// * `true` if the `Mass` was added.
    /// * `false` if the `Mass` was already present.
    pub fn add_mass<C: Cluster<U>>(&mut self, c: &C, position: [f32; DIM]) -> bool {
        // Create the `Mass`
        let m = {
            let mut m = Mass::<DIM>::from_cluster(c);
            m.set_position(position);
            m
        };

        // Add the `Mass`
        self.masses.insert(m.hash_key(), m).is_none()
    }

    /// Remove a `Mass` from the `System`. This will also remove any `Spring`s
    /// connected to the `Mass`.
    ///
    /// # Arguments
    ///
    /// - `offset`: The offset of the `Mass`.
    /// - `cardinality`: The cardinality of the `Mass`.
    ///
    /// # Returns
    ///
    /// * `true` if the `Mass` was removed.
    /// * `false` if the `Mass` was not present.
    pub fn remove_mass(&mut self, offset: usize, cardinality: usize) -> bool {
        let contains_mass = self.masses.contains_key(&(offset, cardinality));

        if contains_mass {
            // Remove the `Spring`s connected to the `Mass`
            self.springs
                .retain(|&(i, j), _| i != (offset, cardinality) && j != (offset, cardinality));
            // Remove the `Mass`
            self.masses.remove(&(offset, cardinality));
        }

        contains_mass
    }

    /// Get the pairs of `Mass`es that are connected by `Spring`s.
    #[must_use]
    pub fn mass_pairs(&self) -> Vec<[&Mass<DIM>; 2]> {
        self.springs
            .iter()
            .map(|((i, j), _)| [&self.masses[i], &self.masses[j]])
            .collect()
    }

    /// For a given mass (by its key), get the keys of the masses it is
    /// connected to and the `Spring`s connecting them.
    #[must_use]
    pub fn connected_masses(&self, i: (usize, usize)) -> Vec<((usize, usize), &Spring<U, DIM>)> {
        self.springs
            .par_iter()
            .filter_map(|(&(m1, m2), s)| {
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

    /// Get the `Spring`s in the `System`.
    #[must_use]
    pub const fn springs(&self) -> &SpringMap<U, DIM> {
        &self.springs
    }

    /// Get the `Spring`s in the `System` as mutable.
    #[must_use]
    pub fn springs_mut(&mut self) -> &mut SpringMap<U, DIM> {
        &mut self.springs
    }

    /// Get a single `Spring` by the keys of the `Mass`es it connects.
    #[must_use]
    pub fn get_spring(&self, i: (usize, usize), j: (usize, usize)) -> Option<&Spring<U, DIM>> {
        self.springs.get(&(i, j))
    }

    /// Get a single `Spring` by the keys of the `Mass`es it connects as mutable.
    #[must_use]
    pub fn get_spring_mut(&mut self, i: (usize, usize), j: (usize, usize)) -> Option<&mut Spring<U, DIM>> {
        self.springs.get_mut(&(i, j))
    }

    /// Add a `Spring` between two `Mass`es.
    ///
    /// # Arguments
    ///
    /// - `i`: The key of the first `Mass`.
    /// - `j`: The key of the second `Mass`.
    /// - `k`: The spring constant of the `Spring`.
    /// - `data`: The `Dataset` containing the `Instance`s.
    ///
    /// # Returns
    ///
    /// * `true` if the `Spring` was added.
    /// * `false` if the `Spring` was already present, in which case the spring
    ///  constant is updated.
    pub fn add_spring<I: Instance, D: Dataset<I, U>>(
        &mut self,
        i: (usize, usize),
        j: (usize, usize),
        k: f32,
        data: &D,
    ) -> bool {
        let l0 = data.one_to_one(self.masses[&i].arg_center(), self.masses[&j].arg_center());
        let mut s = Spring::new(i, j, k, l0);
        let k = s.hash_key();
        s.update_length(&self.masses);
        self.springs.insert(k, s).is_none()
    }

    /// Remove a `Spring` between two `Mass`es.
    ///
    /// # Arguments
    ///
    /// - `i`: The key of the first `Mass`.
    /// - `j`: The key of the second `Mass`.
    ///
    /// # Returns
    ///
    /// * `true` if the `Spring` was removed.
    /// * `false` if the `Spring` was not present.
    pub fn remove_spring(&mut self, i: (usize, usize), j: (usize, usize)) -> bool {
        self.springs.remove(&(i, j)).is_some()
    }

    /// Update the lengths of the `Spring`s and the forces exerted by them.
    #[must_use]
    pub fn update_springs(mut self) -> Self {
        self.springs = self
            .springs
            .into_par_iter()
            .map(|(k, mut s)| {
                s.update_length(&self.masses);
                (k, s)
            })
            .collect();

        self
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
        let forces = self
            .springs
            .par_iter()
            .map(|(&(i, j), s)| {
                let unit_vector = self.masses[&i].unit_vector_to(&self.masses[&j]);
                let f_mag = {
                    // TODO: In mnist data, we somehow hit infinite forces. So, for now,
                    // we are setting the force to 1.0 if it is infinite.
                    if s.f().is_finite() {
                        s.f()
                    } else {
                        1.0
                    }
                };
                let mut force = [0.0; DIM];
                for (f, &uv) in force.iter_mut().zip(unit_vector.iter()) {
                    *f = f_mag * uv;
                }
                (i, j, force)
            })
            .collect::<Vec<_>>();

        // Accumulate the forces for each mass
        for (i, j, force) in forces {
            if let Some(m) = self.masses.get_mut(&i) {
                m.add_force(force);
            }
            if let Some(m) = self.masses.get_mut(&j) {
                m.sub_force(force);
            }
        }

        // Apply the forces to the masses
        self.masses
            .par_iter_mut()
            .for_each(|(_, m)| m.apply_force(dt, self.beta));

        // Update the springs
        // TODO: Merge this iteration with the first one
        self.springs = self
            .springs
            .into_par_iter()
            .map(|(k, mut s)| {
                s.update_length(&self.masses);
                (k, s)
            })
            .collect();

        self
    }

    /// Simulate the `System` until it reaches a stable state.
    ///
    /// # Arguments
    ///
    /// - `dt`: The time-step.
    /// - `patience`: The number of time-steps to consider for stability.
    #[must_use]
    pub fn evolve_to_stability(mut self, dt: f32, patience: usize) -> Self {
        // self.update_logs();

        let mut i = 0;
        let mut stability = self.stability(patience);
        while stability < 0.995 {
            mt_log!(Level::Debug, "Step {i}, Stability: {stability:.6}");
            self = self.update_step(dt);
            i += 1;
            stability = self.stability(patience);
        }

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
        // self.update_logs();

        for i in 0..steps {
            mt_log!(Level::Debug, "Step {i}/{steps}");
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
        self.update_logs(data);
        for i in 0..steps {
            if i % save_every == 0 {
                mt_log!(Level::Debug, "{name}: Saving step {i}/{steps}");

                self.get_reduced_embedding(data, name)
                    .to_npy(&path.join(format!("{i}.npy")))?;
            }
            self = self.update_step(dt);
            self.update_logs(data);
        }

        Ok(self)
    }

    /// Update the `logs`
    fn update_logs<I: Instance, D: Dataset<I, U>>(&mut self, data: &D) {
        let kinetic_energy = self.kinetic_energy();
        let potential_energy = self.potential_energy();
        let total_energy = kinetic_energy + potential_energy;
        let test_callbacks = [
            triangle_tests::are_triangles_equivalent,
            triangle_tests::calc_edge_distortion,
            // triangle_helpers::calc_angle_distortion,
        ];

        if let Some([triangle_equivalence, edge_accuracy]) = triangle_accuracy_tests(self.masses(), data, test_callbacks)
        {
            // self.logs.push(triangle_results);
            self.logs.push([
                kinetic_energy,
                potential_energy,
                total_energy,
                triangle_equivalence,
                edge_accuracy,
                // angle_accuracy,
            ]);
        } else {
            mt_log!(Level::Error, "Triangle results are none");
            self.logs.push([kinetic_energy, potential_energy, total_energy, 0., 0.]);
        }

        // self.logs.push([kinetic_energy, potential_energy, total_energy]);
    }

    /// Get the logs of the `System` for each time-step.
    #[must_use]
    pub fn logs(&self) -> &[LogData] {
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
            let mut masses = self.masses.iter().map(|(&k, m)| (k, m.clone())).collect::<Vec<_>>();
            masses.sort_by(|(a, _), (b, _)| a.cmp(b));
            masses
        };

        let positions = {
            let positions = masses
                .par_iter()
                .flat_map(|((_, c), m)| (0..(*c)).map(move |_| m.position()).collect::<Vec<_>>())
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
        self.springs.par_iter().map(|(_, s)| s.potential_energy()).sum()
    }

    /// Get the total kinetic energy of the `System`.
    #[must_use]
    pub fn kinetic_energy(&self) -> f32 {
        self.masses.par_iter().map(|(_, m)| m.kinetic_energy()).sum()
    }

    /// Get the total energy of the `System`.
    #[must_use]
    pub fn total_energy(&self) -> f32 {
        self.potential_energy() + self.kinetic_energy()
    }

    /// Get the stability of the `System` over the last `n` time-steps.
    ///
    /// The stability is calculated as the mean of the `1 - (std-dev / mean_val)`
    /// of the kinetic and potential energies.
    ///
    /// # Arguments
    ///
    /// - `n`: The number of time-steps to consider.
    ///
    /// # Returns
    ///
    /// The stability of the `System` in a [0, 1] range, with 1 being stable.
    #[must_use]
    #[allow(clippy::similar_names)]
    pub fn stability(&self, n: usize) -> f32 {
        if self.logs.len() < n {
            0.0
        } else {
            let last_n = &self.logs[(self.logs.len() - n)..];

            let (last_ke, last_pe) = last_n
                .iter()
                .map(|&[ke, pe, _, _, _]| (ke, pe))
                .unzip::<_, _, Vec<_>, Vec<_>>();

            let stability_ke = {
                let mean = crate::utils::mean::<_, f32>(&last_ke);
                let variance = crate::utils::variance(&last_ke, mean);
                1.0 - (variance.sqrt() / mean)
            };

            let stability_pe = {
                let mean = crate::utils::mean::<_, f32>(&last_pe);
                let variance = crate::utils::variance(&last_pe, mean);
                1.0 - (variance.sqrt() / mean)
            };

            (stability_ke + stability_pe) / 2.0
        }
    }
}
