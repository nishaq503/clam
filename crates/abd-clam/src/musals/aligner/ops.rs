//! Alignment operations for Needleman-Wunsch algorithm.

/// The direction of the edit operation in the DP table.
#[derive(Clone, Eq, PartialEq, Debug)]
#[must_use]
pub enum Direction {
    /// Diagonal (Up and Left) for a match or substitution.
    Diagonal,
    /// Up for a gap in the first sequence.
    Up,
    /// Left for a gap in the second sequence.
    Left,
}

/// The type of edit operation.
#[must_use]
pub enum Edit {
    /// Substitution of one character for another.
    Sub(u8),
    /// Insertion of a character.
    Ins(u8),
    /// Deletion of a character.
    Del,
}

impl core::fmt::Debug for Edit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sub(c) => f.debug_tuple("Sub").field(&(*c as char)).finish(),
            Self::Ins(c) => f.debug_tuple("Ins").field(&(*c as char)).finish(),
            Self::Del => write!(f, "Del"),
        }
    }
}

/// The sequence of edits needed to turn one unaligned sequence into another.
#[derive(Debug)]
#[must_use]
pub struct Edits(Vec<(usize, Edit)>);

impl Edits {
    /// Create a new `Edits` from a vector.
    pub const fn new(edits: Vec<(usize, Edit)>) -> Self {
        Self(edits)
    }
}

impl core::ops::Deref for Edits {
    type Target = Vec<(usize, Edit)>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
