//! Methods for traversing a `Cluster` tree.

use rayon::prelude::*;

use super::Cluster;

/// Traversal methods for `Cluster` trees.
impl<T, A> Cluster<T, A> {
    /// Traverses the cluster tree in pre-order to apply a function to each cluster.
    ///
    /// WARNING: This may lead to stack overflow for very deep trees.
    pub fn apply_preorder<F, Args>(&mut self, func: &F, args: &Args)
    where
        F: Fn(&mut Self, &Args),
    {
        func(self, args);
        if let Some((children, _, _)) = &mut self.children {
            for child in children.iter_mut() {
                child.apply_preorder(func, args);
            }
        }
    }

    /// Traverses the cluster tree in pre-order to apply a mutable function to each cluster.
    ///
    /// WARNING: This may lead to stack overflow for very deep trees.
    pub fn apply_preorder_mut<F, Args>(&mut self, func: &mut F, args: &mut Args)
    where
        F: FnMut(&mut Self, &mut Args),
    {
        func(self, args);
        if let Some((children, _, _)) = &mut self.children {
            for child in children.iter_mut() {
                child.apply_preorder_mut(func, args);
            }
        }
    }

    /// Traverses the cluster tree in post-order to apply a function to each cluster.
    ///
    /// WARNING: This may lead to stack overflow for very deep trees.
    pub fn apply_postorder<F, Args>(&mut self, func: &F, args: &Args)
    where
        F: Fn(&mut Self, &Args),
    {
        if let Some((children, _, _)) = &mut self.children {
            for child in children.iter_mut() {
                child.apply_postorder(func, args);
            }
        }
        func(self, args);
    }

    /// Traverses the cluster tree in post-order to apply a mutable function to each cluster.
    ///
    /// WARNING: This may lead to stack overflow for very deep trees.
    pub fn apply_postorder_mut<F, Args>(&mut self, func: &mut F, args: &mut Args)
    where
        F: FnMut(&mut Self, &mut Args),
    {
        if let Some((children, _, _)) = &mut self.children {
            for child in children.iter_mut() {
                child.apply_postorder_mut(func, args);
            }
        }
        func(self, args);
    }

    /// Returns all clusters in a stack in post-order.
    pub fn as_postorder_stack(&self) -> Vec<&Self> {
        let mut stack_1 = vec![self];
        let mut stack_2 = Vec::new();

        while let Some(c) = stack_1.pop() {
            if let Some(children) = c.children() {
                stack_1.extend(children);
            }
            stack_2.push(c);
        }

        stack_2
    }

    /// Returns all clusters as leaves in a stack in post-order, placing the span of each cluster alongside it.
    pub fn as_postorder_stack_owned(self) -> Vec<(Self, Option<(Box<[usize]>, T)>)> {
        let mut stack_1 = vec![self];
        let mut stack_2 = Vec::new();

        while let Some(mut c) = stack_1.pop() {
            let cci_and_span = if let Some((children, child_center_indices, span)) = c.take_children_and_span() {
                stack_1.extend(children);
                Some((child_center_indices, span))
            } else {
                None
            };
            stack_2.push((c, cci_and_span));
        }

        stack_2
    }
}

/// Parallel traversal methods for `Cluster` trees.
impl<T, A> Cluster<T, A>
where
    T: Send,
    A: Send,
{
    /// Parallel version of [`Self::apply_preorder`].
    ///
    /// WARNING: This may lead to stack overflow for very deep trees.
    pub fn par_apply_preorder<F, Args>(&mut self, func: &F, args: &Args)
    where
        F: Fn(&mut Self, &Args) + Sync,
        Args: Sync,
    {
        func(self, args);
        if let Some((children, _, _)) = &mut self.children {
            children.par_iter_mut().for_each(|child| {
                child.par_apply_preorder(func, args);
            });
        }
    }

    /// Parallel version of [`Self::apply_postorder`].
    ///
    /// WARNING: This may lead to stack overflow for very deep trees.
    pub fn par_apply_postorder<F, Args>(&mut self, func: &F, args: &Args)
    where
        F: Fn(&mut Self, &Args) + Sync,
        Args: Sync,
    {
        if let Some((children, _, _)) = &mut self.children {
            children.par_iter_mut().for_each(|child| {
                child.par_apply_postorder(func, args);
            });
        }
        func(self, args);
    }
}

/// A `Cluster` may be annotated with the items it contains for intermediate computations in various applications.
pub struct AnnotatedItems<Id, I, A> {
    /// The center item of the cluster.
    pub center: (Id, I),
    /// The other items in the cluster but not in the children or descendants.
    pub non_center: Option<Vec<(Id, I)>>,
    /// The original annotation of the cluster.
    pub annotation: A,
}

impl<T, A> Cluster<T, A> {
    /// Annotates the cluster and its descendants with the items they contain.
    ///
    /// # Panics
    ///
    /// - If `items` is empty.
    pub fn annotate_with_items<Id, I>(self, mut items: Vec<(Id, I)>) -> Cluster<T, AnnotatedItems<Id, I, A>> {
        // TODO(Najib): Do this iteratively to avoid stack overflow for very deep trees.

        // Remove the 0th item as the center and collect the rest if any.
        let mut non_center_items = items.split_off(1);
        let center = items.pop().unwrap_or_else(|| unreachable!("items cannot be empty"));

        if let Some((children, child_center_indices, span)) = self.children {
            let child_items = {
                let mut child_items = Vec::with_capacity(children.len());
                for child in children.iter().rev() {
                    let n = non_center_items.len() - child.cardinality;
                    let c_items = non_center_items.split_off(n);
                    child_items.push(c_items);
                }
                child_items.reverse();
                child_items
            };
            let children = children
                .into_iter()
                .zip(child_items)
                .map(|(child, items)| child.annotate_with_items(items))
                .collect::<Vec<_>>()
                .into_boxed_slice();
            let annotation = AnnotatedItems {
                center,
                non_center: None,
                annotation: self.annotation,
            };
            Cluster {
                depth: self.depth,
                center_index: self.center_index,
                cardinality: self.cardinality,
                radius: self.radius,
                lfd: self.lfd,
                children: Some((children, child_center_indices, span)),
                annotation,
            }
        } else {
            let non_center = if non_center_items.is_empty() { None } else { Some(non_center_items) };
            let annotation = AnnotatedItems {
                center,
                non_center,
                annotation: self.annotation,
            };
            Cluster {
                depth: self.depth,
                center_index: self.center_index,
                cardinality: self.cardinality,
                radius: self.radius,
                lfd: self.lfd,
                children: None,
                annotation,
            }
        }
    }
}

impl<Id, I, T, A> Cluster<T, AnnotatedItems<Id, I, A>> {
    /// Collects all items contained in the cluster and its descendants.
    #[expect(clippy::missing_panics_doc)]
    pub fn collect_items_from_annotations(self) -> (Cluster<T, A>, Vec<(Id, I)>) {
        // TODO(Najib): Do this iteratively to avoid stack overflow for very deep trees.

        let mut items = Vec::with_capacity(self.cardinality);
        let AnnotatedItems {
            center,
            non_center,
            annotation,
        } = self.annotation;
        items.push(center);

        let cluster = if let Some((children, child_center_indices, span)) = self.children {
            let children = children
                .into_iter()
                .map(|child| {
                    let (child, mut child_items) = child.collect_items_from_annotations();
                    items.append(&mut child_items);
                    child
                })
                .collect::<Vec<_>>()
                .into_boxed_slice();

            // TODO(Najib): Remove this assertion after thorough testing.
            assert_eq!(items.len(), self.cardinality, "Collected items do not match cluster cardinality");

            Cluster {
                depth: self.depth,
                center_index: self.center_index,
                cardinality: self.cardinality,
                radius: self.radius,
                lfd: self.lfd,
                children: Some((children, child_center_indices, span)),
                annotation,
            }
        } else {
            if let Some(mut nc_items) = non_center {
                items.append(&mut nc_items);
            }

            // TODO(Najib): Remove this assertion after thorough testing.
            assert_eq!(items.len(), self.cardinality, "Collected items do not match cluster cardinality");

            Cluster {
                depth: self.depth,
                center_index: self.center_index,
                cardinality: self.cardinality,
                radius: self.radius,
                lfd: self.lfd,
                children: None,
                annotation,
            }
        };

        (cluster, items)
    }
}

impl<T, A> Cluster<T, A>
where
    T: Send,
    A: Send,
{
    /// Parallel version of [`Self::annotate_with_items`].
    ///
    /// # Panics
    ///
    /// - If `items` is empty.
    pub fn par_annotate_with_items<Id, I>(self, mut items: Vec<(Id, I)>) -> Cluster<T, AnnotatedItems<Id, I, A>>
    where
        Id: Send,
        I: Send,
    {
        // TODO(Najib): Do this iteratively to avoid stack overflow for very deep trees.

        // Remove the 0th item as the center and collect the rest if any.
        let mut non_center_items = items.split_off(1);
        let center = items.pop().unwrap_or_else(|| unreachable!("items cannot be empty"));

        if let Some((children, child_center_indices, span)) = self.children {
            let child_items = {
                let mut child_items = Vec::with_capacity(children.len());
                for child in children.iter().rev() {
                    let n = non_center_items.len() - child.cardinality;
                    let c_items = non_center_items.split_off(n);
                    child_items.push(c_items);
                }
                child_items.reverse();
                child_items
            };
            let children = children
                .into_par_iter()
                .zip(child_items)
                .map(|(child, items)| child.par_annotate_with_items(items))
                .collect::<Vec<_>>()
                .into_boxed_slice();
            let annotation = AnnotatedItems {
                center,
                non_center: None,
                annotation: self.annotation,
            };
            Cluster {
                depth: self.depth,
                center_index: self.center_index,
                cardinality: self.cardinality,
                radius: self.radius,
                lfd: self.lfd,
                children: Some((children, child_center_indices, span)),
                annotation,
            }
        } else {
            let non_center = if non_center_items.is_empty() { None } else { Some(non_center_items) };
            let annotation = AnnotatedItems {
                center,
                non_center,
                annotation: self.annotation,
            };
            Cluster {
                depth: self.depth,
                center_index: self.center_index,
                cardinality: self.cardinality,
                radius: self.radius,
                lfd: self.lfd,
                children: None,
                annotation,
            }
        }
    }
}

impl<Id, I, T, A> Cluster<T, AnnotatedItems<Id, I, A>>
where
    Id: Send,
    I: Send,
    T: Send,
    A: Send,
{
    /// Parallel version of [`Self::collect_items_from_annotations`].
    #[expect(clippy::missing_panics_doc)]
    pub fn par_collect_items_from_annotations(self) -> (Cluster<T, A>, Vec<(Id, I)>) {
        // TODO(Najib): Do this iteratively to avoid stack overflow for very deep trees.

        let mut items = Vec::with_capacity(self.cardinality);
        let AnnotatedItems {
            center,
            non_center,
            annotation,
        } = self.annotation;
        items.push(center);

        let cluster = if let Some((children, child_center_indices, span)) = self.children {
            let children_and_items = children.into_par_iter().map(Self::par_collect_items_from_annotations).collect::<Vec<_>>();
            let children = children_and_items
                .into_iter()
                .map(|(child, mut child_items)| {
                    items.append(&mut child_items);
                    child
                })
                .collect::<Vec<_>>()
                .into_boxed_slice();

            // TODO(Najib): Remove this assertion after thorough testing.
            assert_eq!(items.len(), self.cardinality, "Collected items do not match cluster cardinality");

            Cluster {
                depth: self.depth,
                center_index: self.center_index,
                cardinality: self.cardinality,
                radius: self.radius,
                lfd: self.lfd,
                children: Some((children, child_center_indices, span)),
                annotation,
            }
        } else {
            if let Some(mut nc_items) = non_center {
                items.append(&mut nc_items);
            }

            // TODO(Najib): Remove this assertion after thorough testing.
            assert_eq!(items.len(), self.cardinality, "Collected items do not match cluster cardinality");

            Cluster {
                depth: self.depth,
                center_index: self.center_index,
                cardinality: self.cardinality,
                radius: self.radius,
                lfd: self.lfd,
                children: None,
                annotation,
            }
        };

        (cluster, items)
    }
}
