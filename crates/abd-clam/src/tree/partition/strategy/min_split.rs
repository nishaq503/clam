//! The minimum fraction of items that must be in the smaller child cluster when partitioning a cluster.

/// The minimum fraction of items that must be in the smaller child cluster when partitioning a cluster.
#[must_use]
#[derive(Debug, Clone, Copy, Default)]
pub enum MinSplit {
    /// The minimum fraction of items in the smaller child cluster is fixed.
    Fixed(f64),
    /// The minimum fraction of items in the smaller child cluster is `1 / 10`.
    Tenth,
    /// The minimum fraction of items in the smaller child cluster is `1 / 4`.
    Quarter,
    /// No minimum fraction is enforced. This is the default.
    #[default]
    None,
}

impl MinSplit {
    /// Returns the maximum number of items that can be in the larger child cluster when partitioning a cluster of the given cardinality.
    #[must_use]
    #[expect(clippy::cast_sign_loss, clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn max_items_for(&self, cardinality: usize) -> Option<usize> {
        let fraction = match self {
            Self::Fixed(fraction) => Some(*fraction),
            Self::Tenth => Some(0.1),
            Self::Quarter => Some(0.25),
            Self::None => None,
        };
        fraction.map(|f| ((cardinality as f64) * (1.0 - f)).floor() as usize).map(|n| n.max(1))
    }
}
