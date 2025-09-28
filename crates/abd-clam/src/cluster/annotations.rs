//! `Cluster`s can have arbitrary annotations associated with them.

use crate::DistanceValue;

use super::{Cluster, Contents};

impl<Id, I, T: DistanceValue, A> Cluster<Id, I, T, A> {
    /// Changes the type of annotations in the tree, setting all annotations to `None`.
    pub fn reset_annotations<B>(self) -> Cluster<Id, I, T, B> {
        let contents = match self.contents {
            Contents::Leaf(items) => Contents::Leaf(items),
            Contents::Children([left, right]) => {
                Contents::Children([Box::new(left.reset_annotations()), Box::new(right.reset_annotations())])
            }
        };
        Cluster {
            cardinality: self.cardinality,
            center: self.center,
            radius: self.radius,
            lfd: self.lfd,
            radial_sum: self.radial_sum,
            contents,
            annotation: None,
        }
    }

    /// A mutable reference to the annotations, if any.
    pub const fn annotation_mut(&mut self) -> Option<&mut A> {
        self.annotation.as_mut()
    }

    /// Takes the annotations, leaving `None` in its place.
    pub const fn take_annotation(&mut self) -> Option<A> {
        self.annotation.take()
    }

    /// Annotates all clusters in the tree by applying the provided function in a pre-order traversal.
    pub fn annotate_pre_order<Pre: Fn(&Self) -> A>(&mut self, pre: &Pre) {
        self.annotation = Some(pre(self));
        if let Contents::Children([left, right]) = &mut self.contents {
            left.annotate_pre_order(pre);
            right.annotate_pre_order(pre);
        }
    }

    /// Annotates all clusters in the tree by applying the provided function in a post-order traversal.
    pub fn annotate_post_order<Post: Fn(&Self) -> A>(&mut self, post: &Post) {
        self.annotation = Some(post(self));
        if let Contents::Children([left, right]) = &mut self.contents {
            left.annotate_post_order(post);
            right.annotate_post_order(post);
        }
    }

    /// Annotates all clusters in the tree first by applying the `pre` function before visiting children, and then applying the `post` function after visiting the
    /// children, in a full traversal.
    pub fn annotate_pre_post<Pre: Fn(&Self) -> A, Post: Fn(&Self) -> A>(&mut self, pre: &Pre, post: &Post) {
        self.annotation = Some(pre(self));
        if let Contents::Children([left, right]) = &mut self.contents {
            left.annotate_pre_post(pre, post);
            right.annotate_pre_post(pre, post);
        }
        self.annotation = Some(post(self));
    }

    /// Same as [`annotate_pre_order`](Self::annotate_pre_order) but the function can mutate its internal state.
    pub fn annotate_pre_order_mut<Pre: FnMut(&Self) -> A>(&mut self, pre: &mut Pre) {
        self.annotation = Some(pre(self));
        if let Contents::Children([left, right]) = &mut self.contents {
            left.annotate_pre_order_mut(pre);
            right.annotate_pre_order_mut(pre);
        }
    }

    /// Same as [`annotate_post_order`](Self::annotate_post_order) but the function can mutate its internal state.
    pub fn annotate_post_order_mut<Post: FnMut(&Self) -> A>(&mut self, post: &mut Post) {
        self.annotation = Some(post(self));
        if let Contents::Children([left, right]) = &mut self.contents {
            left.annotate_post_order_mut(post);
            right.annotate_post_order_mut(post);
        }
    }

    /// Same as [`annotate_pre_post`](Self::annotate_pre_post) but the functions can mutate their internal state.
    pub fn annotate_pre_post_mut<Pre: FnMut(&Self) -> A, Post: FnMut(&Self) -> A>(
        &mut self,
        pre: &mut Pre,
        post: &mut Post,
    ) {
        self.annotation = Some(pre(self));
        if let Contents::Children([left, right]) = &mut self.contents {
            left.annotate_pre_post_mut(pre, post);
            right.annotate_pre_post_mut(pre, post);
        }
        self.annotation = Some(post(self));
    }
}

impl<Id: Send + Sync, I: Send + Sync, T: DistanceValue + Send + Sync, A: Send + Sync> Cluster<Id, I, T, A> {
    /// Parallel version of [`reset_annotations`](Self::reset_annotations).
    pub fn par_reset_annotations<B: Send + Sync>(self) -> Cluster<Id, I, T, B> {
        let contents = match self.contents {
            Contents::Leaf(items) => Contents::Leaf(items),
            Contents::Children([left, right]) => {
                let (left, right) = rayon::join(|| left.par_reset_annotations(), || right.par_reset_annotations());
                Contents::Children([Box::new(left), Box::new(right)])
            }
        };
        Cluster {
            cardinality: self.cardinality,
            center: self.center,
            radius: self.radius,
            lfd: self.lfd,
            radial_sum: self.radial_sum,
            contents,
            annotation: None,
        }
    }

    /// Parallel version of [`annotate_pre_order`](Self::annotate_pre_order).
    pub fn par_annotate_pre_order<Pre: Fn(&Self) -> A + Send + Sync>(&mut self, pre: &Pre) {
        self.annotation = Some(pre(self));
        if let Contents::Children([left, right]) = &mut self.contents {
            rayon::join(
                || left.par_annotate_pre_order(pre),
                || right.par_annotate_pre_order(pre),
            );
        }
    }

    /// Parallel version of [`annotate_post_order`](Self::annotate_post_order).
    pub fn par_annotate_post_order<Post: Fn(&Self) -> A + Send + Sync>(&mut self, post: &Post) {
        self.annotation = Some(post(self));
        if let Contents::Children([left, right]) = &mut self.contents {
            rayon::join(
                || left.par_annotate_post_order(post),
                || right.par_annotate_post_order(post),
            );
        }
    }

    /// Parallel version of [`annotate_pre_post`](Self::annotate_pre_post).
    pub fn par_annotate_pre_post<Pre: Fn(&Self) -> A + Send + Sync, Post: Fn(&Self) -> A + Send + Sync>(
        &mut self,
        pre: &Pre,
        post: &Post,
    ) {
        self.annotation = Some(pre(self));
        if let Contents::Children([left, right]) = &mut self.contents {
            rayon::join(
                || left.par_annotate_pre_post(pre, post),
                || right.par_annotate_pre_post(pre, post),
            );
        }
        self.annotation = Some(post(self));
    }
}
