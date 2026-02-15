//! Compression and decompression of trees with items implementing the `Codec` trait.

use std::collections::HashMap;

use crate::{Cluster, DistanceValue, Tree};

use super::{Codec, MaybeCompressed};

mod par_tree;

/// A cluster in the frontier of the compression algorithm, containing compressed items and waiting for its siblings and parent to be compressed.
#[derive(Debug)]
pub struct FrontierCluster<I, T, A>
where
    I: Codec,
{
    /// The index of the center of this cluster in the items vector.
    pub id: usize,
    /// The index of the center of the parent cluster in the items vector, if the cluster is not the root.
    pub pid: Option<usize>,
    /// The cost of unitary/recursive compression of this cluster.
    pub cost: usize,
    /// The cluster itself, annotated with the compressed items and the old annotation.
    pub cluster: Cluster<T, (usize, Vec<I::Compressed>, A)>,
    /// Whether this cluster is recursively compressed or not.
    pub is_recursive: bool,
}

impl<I, T, A> FrontierCluster<I, T, A>
where
    I: Codec,
{
    /// Creates a new frontier cluster using unitary compression.
    fn unitary<Id>(mut cluster: Cluster<T, A>, items: &[(Id, I)]) -> Self
    where
        T: DistanceValue,
    {
        let id = cluster.center_index;
        let pid = cluster.parent_center_index;
        let (cost, compressed_items) = unitary_annotator(&cluster, items);

        // SAFETY: We own the cluster and will replace its annotation before we return. This trick allows us to avoid requiring `A: Default` as a trait bound.
        #[expect(unsafe_code, clippy::mem_replace_with_uninit)]
        let old_annotation = unsafe { core::mem::replace(&mut cluster.annotation, core::mem::zeroed()) };
        let cluster = cluster.change_annotation_with(|_, _, ()| (cost, compressed_items, old_annotation), ());

        Self {
            id,
            pid,
            cost,
            cluster,
            is_recursive: false,
        }
    }

    /// Creates a new frontier cluster using recursive compression.
    fn recursive<Id>(&self, items: &[(Id, I)], children: &[Self]) -> (usize, Vec<I::Compressed>)
    where
        T: DistanceValue,
    {
        // Compress the center of each child cluster in terms of the center of the current cluster.
        let center = &items[self.id].1;
        let (child_costs, child_centers): (Vec<_>, Vec<_>) = children
            .iter()
            .map(|child| {
                let child_center = center.compress(&items[child.id].1);
                let child_cost = child.cluster.annotation().0 + I::compressed_size(&child_center);
                (child_cost, child_center)
            })
            .unzip();
        // Compute the cost of recursive compression.
        let cost = child_costs.into_iter().sum::<usize>();
        (cost, child_centers)
    }
}

/// Applies unitary compression to the items in the cluster, and returns the cost of compression and the compressed items.
fn unitary_annotator<Id, I, T, A>(cluster: &Cluster<T, A>, items: &[(Id, I)]) -> (usize, Vec<I::Compressed>)
where
    I: Codec,
    T: DistanceValue,
{
    let center = &items[cluster.center_index].1;
    let (costs, items): (Vec<_>, Vec<_>) = items[cluster.subtree_indices()]
        .iter()
        .map(|(_, item)| {
            let item = center.compress(item);
            let cost = I::compressed_size(&item);
            (cost, item)
        })
        .unzip();
    let cost = costs.into_iter().sum();
    (cost, items)
}

impl<Id, I, T, A, M> Tree<Id, I, T, A, M>
where
    I: Codec,
{
    /// Returns the tree with compressed items.
    #[expect(clippy::missing_panics_doc)]
    pub fn compress_all(self, min_depth: usize) -> Tree<Id, MaybeCompressed<I>, T, A, M>
    where
        T: DistanceValue,
    {
        let (items, cluster_map, metric) = self.into_parts();
        let n_clusters = cluster_map.len();

        // Apply unitary compression to all clusters, annotating them with their compressed items, and partition them into the frontier (leaf clusters) and the
        // parents (non-leaf clusters).
        let (mut frontier, parents): (Vec<_>, Vec<_>) = cluster_map
            .into_values()
            .map(|cluster| FrontierCluster::unitary(cluster, &items))
            .partition(|cluster| cluster.cluster.is_leaf());

        // Map of parent clusters waiting for their children to be compressed.
        let mut parents_in_waiting = parents
            .into_iter()
            .map(|parent| {
                let id = parent.id;
                let n_children = parent.cluster.children.as_ref().map_or(0, |(child_indices, _)| child_indices.len());
                (id, (n_children, Vec::with_capacity(n_children), parent))
            })
            .collect::<HashMap<_, _>>();

        // Keep all clusters for now.
        let mut all_cluster_map = HashMap::new();

        // Traverse the tree from the frontier to the root, and compress the clusters as we go up.
        while !parents_in_waiting.is_empty() {
            assert!(
                !frontier.is_empty(),
                "There should always be clusters in the frontier while there are parents waiting for their children to be compressed"
            );

            for cluster in frontier {
                // Update the waiting parent of this cluster.
                let pid = cluster.pid.unwrap_or(0);
                let (pending_children, child_compressions, _) = parents_in_waiting.get_mut(&pid).unwrap_or_else(|| {
                    unreachable!(
                        "Parent cluster with index {pid} should be in waiting when processing its child cluster with index {}",
                        cluster.id
                    )
                });

                // Update the pending children count and the compressions of the child cluster.
                *pending_children -= 1;
                child_compressions.push(cluster);
            }

            // Find all parents that have no more pending children
            let full_parents: HashMap<_, _>;
            (full_parents, parents_in_waiting) = parents_in_waiting.into_iter().partition(|(_, (pending_children, _, _))| *pending_children == 0);

            // Apply recursive compression to the full parents and add them to the next frontier.
            let old_frontier: Vec<_>;
            (old_frontier, frontier) = full_parents
                .into_values()
                .map(|(_, mut children, mut parent)| {
                    // Sort the children by their center indices to have them in the same order as the children are stored in the cluster.
                    children.sort_by_key(|child| child.id);
                    // Calculate the recursive compression of the children in terms of the parent.
                    let (rec_cost, child_centers) = parent.recursive(&items, &children);
                    // If the parent is too shallow, or recursive compression is cheaper, then we keep the recursive compression. Otherwise, we keep the unitary
                    // compression.
                    if parent.cluster.depth() <= min_depth || rec_cost < parent.cost {
                        parent.is_recursive = true;
                        let ann = parent.cluster.annotation_mut();
                        ann.0 = rec_cost;
                        ann.1 = child_centers;
                    }
                    (children, parent)
                })
                .unzip();

            // Add the old frontier clusters to the clusters map, since we want to keep them for the Tree that we will return at the end.
            all_cluster_map.extend(old_frontier.into_iter().flatten().map(|cluster| (cluster.id, cluster)));
        }

        assert_eq!(
            frontier.len(),
            1,
            "There should be only one cluster in the frontier at the end, which is the root cluster."
        );
        assert_eq!(
            all_cluster_map.len(),
            n_clusters - 1,
            "All clusters except the root cluster should be in the all_cluster_map."
        );

        // We only have the root cluster left in the frontier. We will now traverse down from the root and stop at the first unitary cluster along each branch.
        let mut cluster_map = HashMap::new();
        let mut items = items.into_iter().map(|(id, item)| (id, MaybeCompressed::Original(item))).collect::<Vec<_>>();

        while let Some(cluster) = frontier.pop() {
            let FrontierCluster { mut cluster, is_recursive, .. } = cluster;

            // Get the indices of the compressed items we need to update for this cluster.
            let indices = if is_recursive {
                // The cluster uses recursive compression, so we need to update the compressed centers of its children.
                let indices = cluster
                    .child_center_indices()
                    .unwrap_or_else(|| unreachable!("A recursively compressed cluster should have child center indices"));

                // Add the children of the cluster to the frontier because they may also be recursively compressed.
                for i in indices {
                    let child = all_cluster_map.remove(i).unwrap_or_else(|| {
                        unreachable!(
                            "Child cluster with index {i} should be in the clusters map when processing its parent cluster with index {}",
                            cluster.center_index
                        )
                    });
                    frontier.push(child);
                }
                indices.to_vec()
            } else {
                // We can remove the children of the cluster because it uses unitary compression.
                cluster.children = None;

                // The cluster uses unitary compression, so we need to update all the items in its subtree.
                cluster.subtree_indices().collect()
            };

            // Get the compressed items from the annotation of the cluster and update the items in the tree with the compressed items.
            let ann = cluster.annotation_mut();
            let compressed_items = core::mem::take(&mut ann.1);
            for (index, compressed_item) in indices.into_iter().zip(compressed_items) {
                items[index].1 = MaybeCompressed::Compressed(compressed_item);
            }

            // SAFETY: We will replace the annotation of the cluster before we return.
            #[expect(unsafe_code, clippy::mem_replace_with_uninit)]
            let annotation = unsafe { core::mem::replace(&mut ann.2, core::mem::zeroed()) };
            let cluster = cluster.change_annotation(annotation);

            cluster_map.insert(cluster.center_index, cluster);
        }

        Tree::from_parts(items, cluster_map, metric)
    }
}

impl<Id, I, T, A, M> Tree<Id, MaybeCompressed<I>, T, A, M>
where
    I: Codec,
{
    /// Returns the tree with decompressed items.
    pub fn decompress_all(mut self) -> Tree<Id, I, T, A, M> {
        self.decompress_subtree(0)
            .unwrap_or_else(|_| unreachable!("The center of the root cluster is never compressed."));
        self.apply_to_items(&|id, item| {
            let item = item
                .take_original()
                .unwrap_or_else(|| unreachable!("All items should be in their original form by the time the frontier is empty"));
            (id, item)
        })
    }

    /// Given the index of a cluster center, decompresses the child centers of that cluster.
    ///
    /// # Arguments
    ///
    /// - `id`: index of the cluster center, which must be decompressed.
    ///
    /// # Returns
    ///
    /// - If the cluster has children, returns the indices of the child centers.
    ///
    /// # Errors
    ///
    /// - If the `id` is not the center of any cluster.
    /// - If the cluster center is compressed.
    pub(crate) fn decompress_child_centers(&mut self, id: usize) -> Result<Option<Vec<usize>>, String> {
        if let Some(targets) = self.get_cluster(id)?.child_center_indices().map(<[_]>::to_vec) {
            let items = self.decompressed_items(id, &targets)?;
            for (i, item) in items.into_iter().flatten() {
                self.items[i].1 = MaybeCompressed::Original(item);
            }
            Ok(Some(targets))
        } else {
            Ok(None)
        }
    }

    /// Given the index of a cluster center, decompresses the subtree of that cluster if the given center is already decompressed, and errors otherwise.
    pub(crate) fn decompress_subtree(&mut self, id: usize) -> Result<(), String> {
        let mut frontier = vec![id];
        while let Some(id) = frontier.pop() {
            if let Some(child_centers) = self.decompress_child_centers(id)? {
                // Add the children of the cluster to the frontier because they may also be recursively compressed.
                frontier.extend(child_centers);
            } else {
                // This is a unitarily compressed cluster, so we need to decompress all the non-center items that are compressed.
                let targets = self.get_cluster(id)?.subtree_indices().collect::<Vec<_>>();
                let dec_items = self.decompressed_items(id, &targets)?;
                for (i, item) in dec_items.into_iter().flatten() {
                    self.items[i].1 = MaybeCompressed::Original(item);
                }
            }
        }

        Ok(())
    }

    /// Returns decompressed versions of the indexed items in terms of the indexed center.
    ///
    /// # Arguments
    ///
    /// - `center`: index of the center, which must currently be decompressed.
    /// - `targets`: indices of the items to decompress.
    ///
    /// # Returns
    ///
    /// A vector of the same length as `targets`, where each element is `Some((index, item))` if the corresponding target was successfully decompressed, and
    /// `None` if the corresponding target was already decompressed.
    ///
    /// # Errors
    ///
    /// - If the indexed center is compressed.
    pub(crate) fn decompressed_items(&self, center: usize, targets: &[usize]) -> Result<Vec<Option<(usize, I)>>, String> {
        let center = self.items[center]
            .1
            .original()
            .ok_or_else(|| format!("Center item at index {center} was compressed"))?;
        Ok(targets
            .iter()
            .map(|&i| self.items[i].1.compressed().map(|compressed| (i, center.decompress(compressed))))
            .collect())
    }
}
