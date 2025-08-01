//! A helpful struct to store a root `Cluster`, a `Dataset`, and a `Metric`.

use distances::Number;

use super::{Cluster, Dataset, Metric, ParCluster, ParDataset, ParMetric};

/// A helpful struct to store a root `Cluster`, a `Dataset`, and a `Metric`.
#[derive(Clone)]
#[cfg_attr(
    feature = "disk-io",
    derive(bitcode::Encode, bitcode::Decode, serde::Deserialize, serde::Serialize)
)]
#[must_use]
pub struct Tree<I, T: Number, D: Dataset<I>, C: Cluster<T>, M: Metric<I, T>> {
    /// The root `Cluster` of the tree.
    root: C,
    /// The `Dataset` of the tree.
    dataset: D,
    /// The `Metric` of the tree.
    metric: M,
    /// To satisfy the compiler.
    phantom: std::marker::PhantomData<(I, T)>,
}

impl<I, T: Number, D: Dataset<I>, C: Cluster<T>, M: Metric<I, T>> Tree<I, T, D, C, M> {
    /// Create a new `Tree` with the given root `Cluster`, `Dataset`, and `Metric`.
    pub const fn new(root: C, dataset: D, metric: M) -> Self {
        Self {
            root,
            dataset,
            metric,
            phantom: std::marker::PhantomData,
        }
    }

    /// Get the root `Cluster` of the tree.
    pub const fn root(&self) -> &C {
        &self.root
    }

    /// Get the `Dataset` of the tree.
    pub const fn dataset(&self) -> &D {
        &self.dataset
    }

    /// Get the `Metric` of the tree.
    pub const fn metric(&self) -> &M {
        &self.metric
    }

    /// Change the root `Cluster` of the tree.
    pub fn set_root(&mut self, root: C) {
        self.root = root;
    }

    /// Change the root `Cluster` of the tree and return the new tree.
    pub fn with_root(self, root: C) -> Self {
        Self { root, ..self }
    }

    /// Change the `Dataset` of the tree.
    pub fn set_dataset(&mut self, dataset: D) {
        self.dataset = dataset;
    }

    /// Change the `Dataset` of the tree and return the new tree.
    pub fn with_dataset(self, dataset: D) -> Self {
        Self { dataset, ..self }
    }

    /// Change the `Metric` of the tree.
    pub fn set_metric(&mut self, metric: M) {
        self.metric = metric;
    }

    /// Change the `Metric` of the tree and return the new tree.
    pub fn with_metric(self, metric: M) -> Self {
        Self { metric, ..self }
    }

    /// Decompose the tree into its root `Cluster`, `Dataset`, and `Metric`.
    pub fn deconstruct(self) -> (C, D, M) {
        (self.root, self.dataset, self.metric)
    }
}

#[cfg(feature = "disk-io")]
impl<I, T, D, C, M> crate::DiskIO for Tree<I, T, D, C, M>
where
    I: bitcode::Encode + bitcode::Decode,
    T: Number + bitcode::Encode + bitcode::Decode,
    D: Dataset<I> + crate::DiskIO,
    C: Cluster<T> + crate::DiskIO,
    M: Metric<I, T> + bitcode::Encode + bitcode::Decode,
{
    fn to_bytes(&self) -> Result<Vec<u8>, String> {
        let root = self.root.to_bytes()?;
        let data = self.dataset.to_bytes()?;
        let metric = bitcode::encode(&self.metric).map_err(|e| e.to_string())?;

        let mut bytes = Vec::with_capacity(root.len() + data.len() + metric.len() + std::mem::size_of::<usize>() * 3);
        bytes.extend_from_slice(&root.len().to_le_bytes());
        bytes.extend_from_slice(&root);
        bytes.extend_from_slice(&data.len().to_le_bytes());
        bytes.extend_from_slice(&data);
        bytes.extend_from_slice(&metric.len().to_le_bytes());
        bytes.extend_from_slice(&metric);

        Ok(bytes)
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut offset = 0;

        let root_bytes = crate::utils::read_encoding(bytes, &mut offset);
        let dataset_bytes = crate::utils::read_encoding(bytes, &mut offset);
        let metric_bytes = crate::utils::read_encoding(bytes, &mut offset);

        let root = C::from_bytes(&root_bytes)?;
        let dataset = D::from_bytes(&dataset_bytes)?;
        let metric = bitcode::decode(&metric_bytes).map_err(|e| e.to_string())?;

        Ok(Self::new(root, dataset, metric))
    }
}

#[cfg(feature = "disk-io")]
impl<I, T, D, C, M> crate::ParDiskIO for Tree<I, T, D, C, M>
where
    I: bitcode::Encode + bitcode::Decode + Send + Sync,
    T: Number + bitcode::Encode + bitcode::Decode + Send + Sync,
    D: ParDataset<I> + crate::ParDiskIO,
    C: ParCluster<T> + crate::ParDiskIO,
    M: ParMetric<I, T> + bitcode::Encode + bitcode::Decode,
{
}
