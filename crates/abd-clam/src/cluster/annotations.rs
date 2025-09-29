//! `Cluster`s can have arbitrary annotations associated with them.

use rayon::prelude::*;

use crate::DistanceValue;

use super::{Cluster, Contents};

impl<Id, I, T: DistanceValue, A> Cluster<Id, I, T, A> {
    /// A wrapper to remove the annotations from the tree and change the type of annotations to the unit type `()`.
    pub fn remove_annotations(self) -> Cluster<Id, I, T, ()> {
        self.change_annotation_type(&|_, _| None)
    }

    /// Changes the type of annotations in the tree, applying the provided function, in post-order, to each cluster and the previous annotation (if any) to
    /// produce the new annotation.
    pub fn change_annotation_type<B, Post: Fn(&Cluster<Id, I, T, B>, Option<A>) -> Option<B>>(
        self,
        post: &Post,
    ) -> Cluster<Id, I, T, B> {
        let contents = match self.contents {
            Contents::Leaf(items) => Contents::Leaf(items),
            Contents::Children(children) => Contents::Children(
                children
                    .into_iter()
                    .map(|child| child.change_annotation_type(post))
                    .map(Box::new)
                    .collect::<Vec<_>>(),
            ),
        };
        let mut c = Cluster {
            cardinality: self.cardinality,
            center: self.center,
            radius: self.radius,
            lfd: self.lfd,
            radial_sum: self.radial_sum,
            span: self.span,
            contents,
            annotation: None,
        };
        c.annotation = post(&c, self.annotation);
        c
    }

    /// Annotates all clusters in the tree by applying the provided function in a pre-order traversal.
    pub fn annotate_pre_order<Pre: Fn(&Self) -> A>(&mut self, pre: &Pre) {
        self.annotation = Some(pre(self));
        if let Contents::Children(children) = &mut self.contents {
            for child in children {
                child.annotate_pre_order(pre);
            }
        }
    }

    /// Annotates all clusters in the tree by applying the provided function in a post-order traversal.
    pub fn annotate_post_order<Post: Fn(&Self) -> A>(&mut self, post: &Post) {
        if let Contents::Children(children) = &mut self.contents {
            for child in children {
                child.annotate_post_order(post);
            }
        }
        self.annotation = Some(post(self));
    }

    /// Annotates all clusters in the tree first by applying the `pre` function before visiting children, and then applying the `post` function after visiting the
    /// children, in a full traversal.
    pub fn annotate_pre_post<Pre: Fn(&Self) -> A, Post: Fn(&Self) -> A>(&mut self, pre: &Pre, post: &Post) {
        self.annotation = Some(pre(self));
        if let Contents::Children(children) = &mut self.contents {
            for child in children {
                child.annotate_pre_post(pre, post);
            }
        }
        self.annotation = Some(post(self));
    }

    /// Same as [`annotate_pre_order`](Self::annotate_pre_order) but the function can mutate its internal state.
    pub fn annotate_pre_order_mut<Pre: FnMut(&Self) -> A>(&mut self, pre: &mut Pre) {
        self.annotation = Some(pre(self));
        if let Contents::Children(children) = &mut self.contents {
            for child in children {
                child.annotate_pre_order_mut(pre);
            }
        }
    }

    /// Same as [`annotate_post_order`](Self::annotate_post_order) but the function can mutate its internal state.
    pub fn annotate_post_order_mut<Post: FnMut(&Self) -> A>(&mut self, post: &mut Post) {
        if let Contents::Children(children) = &mut self.contents {
            for child in children {
                child.annotate_post_order_mut(post);
            }
        }
        self.annotation = Some(post(self));
    }

    /// Same as [`annotate_pre_post`](Self::annotate_pre_post) but the functions can mutate their internal state.
    pub fn annotate_pre_post_mut<Pre: FnMut(&Self) -> A, Post: FnMut(&Self) -> A>(
        &mut self,
        pre: &mut Pre,
        post: &mut Post,
    ) {
        self.annotation = Some(pre(self));
        if let Contents::Children(children) = &mut self.contents {
            for child in children {
                child.annotate_pre_post_mut(pre, post);
            }
        }
        self.annotation = Some(post(self));
    }
}

impl<Id: Send + Sync, I: Send + Sync, T: DistanceValue + Send + Sync, A: Send + Sync> Cluster<Id, I, T, A> {
    /// Parallel version of [`remove_annotations`](Self::remove_annotations).
    pub fn par_remove_annotations(self) -> Cluster<Id, I, T, ()> {
        self.change_annotation_type(&|_, _| None)
    }

    /// Parallel version of [`change_annotation_type`](Self::change_annotation_type).
    pub fn par_change_annotation_type<
        B: Send + Sync,
        Post: Fn(&Cluster<Id, I, T, B>, Option<A>) -> Option<B> + Send + Sync,
    >(
        self,
        post: &Post,
    ) -> Cluster<Id, I, T, B> {
        let contents = match self.contents {
            Contents::Leaf(items) => Contents::Leaf(items),
            Contents::Children(children) => Contents::Children(
                children
                    .into_par_iter()
                    .map(|child| child.par_change_annotation_type(post))
                    .map(Box::new)
                    .collect::<Vec<_>>(),
            ),
        };
        let mut c = Cluster {
            cardinality: self.cardinality,
            center: self.center,
            radius: self.radius,
            lfd: self.lfd,
            radial_sum: self.radial_sum,
            span: self.span,
            contents,
            annotation: None,
        };
        c.annotation = post(&c, self.annotation);
        c
    }

    /// Parallel version of [`annotate_pre_order`](Self::annotate_pre_order).
    pub fn par_annotate_pre_order<Pre: Fn(&Self) -> A + Send + Sync>(&mut self, pre: &Pre) {
        self.annotation = Some(pre(self));
        if let Contents::Children(children) = &mut self.contents {
            children
                .par_iter_mut()
                .for_each(|child| child.par_annotate_pre_order(pre));
        }
    }

    /// Parallel version of [`annotate_post_order`](Self::annotate_post_order).
    pub fn par_annotate_post_order<Post: Fn(&Self) -> A + Send + Sync>(&mut self, post: &Post) {
        if let Contents::Children(children) = &mut self.contents {
            children
                .par_iter_mut()
                .for_each(|child| child.par_annotate_post_order(post));
        }
        self.annotation = Some(post(self));
    }

    /// Parallel version of [`annotate_pre_post`](Self::annotate_pre_post).
    pub fn par_annotate_pre_post<Pre: Fn(&Self) -> A + Send + Sync, Post: Fn(&Self) -> A + Send + Sync>(
        &mut self,
        pre: &Pre,
        post: &Post,
    ) {
        self.annotation = Some(pre(self));
        if let Contents::Children(children) = &mut self.contents {
            children
                .par_iter_mut()
                .for_each(|child| child.par_annotate_pre_post(pre, post));
        }
        self.annotation = Some(post(self));
    }
}
