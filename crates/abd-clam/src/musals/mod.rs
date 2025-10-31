//! Multiple Sequence Alignment At Scale (`MuSAlS`) with CLAM.

mod aligner;

pub use aligner::{Aligner, CostMatrix, ops};

/// A trait for sequence types used in `MuSAlS`.
pub trait Sequence: AsRef<[u8]> + Sized {
    /// Creates a sequence from a vector of bytes.
    #[must_use]
    fn from_vec(vec: Vec<u8>) -> Self;

    /// Creates a sequence from a String.
    #[must_use]
    fn from_string(s: String) -> Self {
        Self::from_vec(s.into_bytes())
    }

    /// Applies the given edits to the sequence.
    #[must_use]
    fn apply_edits(&self, edits: &ops::Edits) -> Self {
        let mut result = Vec::with_capacity(self.as_ref().len() + edits.len());
        result.extend(self.as_ref());

        let mut offset = 0;
        for (i, edit) in edits.iter() {
            match edit {
                ops::Edit::Sub(c) => {
                    result[i + offset] = *c;
                }
                ops::Edit::Ins(c) => {
                    result.insert(i + offset, *c);
                    offset += 1;
                }
                ops::Edit::Del => {
                    result.remove(i + offset);
                    offset -= 1;
                }
            }
        }

        Self::from_vec(result)
    }
}

impl Sequence for String {
    #[allow(clippy::expect_used)]
    fn from_vec(vec: Vec<u8>) -> Self {
        Self::from_utf8(vec).expect("Invalid UTF-8 sequence")
    }

    fn from_string(s: String) -> Self {
        s
    }
}

impl Sequence for Vec<u8> {
    fn from_vec(vec: Vec<u8>) -> Self {
        vec
    }
}

impl Sequence for Box<[u8]> {
    fn from_vec(vec: Vec<u8>) -> Self {
        vec.into_boxed_slice()
    }
}
