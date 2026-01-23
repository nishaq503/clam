//! Methods for traversing a `Cluster` tree.

use rayon::prelude::*;

use crate::DistanceValue;

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
        if let Some((children, _)) = &mut self.children {
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
        if let Some((children, _)) = &mut self.children {
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
        if let Some((children, _)) = &mut self.children {
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
        if let Some((children, _)) = &mut self.children {
            for child in children.iter_mut() {
                child.apply_postorder_mut(func, args);
            }
        }
        func(self, args);
    }

    /// Returns all clusters in a stack in post-order.
    pub fn as_postorder_stack(&self) -> Vec<&Self>
    where
        T: DistanceValue,
    {
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
    pub fn as_postorder_stack_owned(self) -> Vec<(Self, Option<T>)> {
        let mut stack_1 = vec![self];
        let mut stack_2 = Vec::new();

        while let Some(mut c) = stack_1.pop() {
            let span = if let Some((children, span)) = c.take_children_and_span() {
                stack_1.extend(children);
                Some(span)
            } else {
                None
            };
            stack_2.push((c, span));
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
        if let Some((children, _)) = &mut self.children {
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
        if let Some((children, _)) = &mut self.children {
            children.par_iter_mut().for_each(|child| {
                child.par_apply_postorder(func, args);
            });
        }
        func(self, args);
    }
}
