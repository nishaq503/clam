//! An item that has been compressed using an `Encoder` and can be decompressed during `CompressiveSearch`.

use super::{Decoder, Encoder};

/// An item that has been compressed using an `Encoder` and can be decompressed during `CompressiveSearch`.
#[derive(Debug, Clone)]
pub enum CodecItem<I, Enc: Encoder<I, Dec> + ?Sized, Dec: Decoder<I, Enc> + ?Sized> {
    /// An uncompressed item.
    Uncompressed(I),
    /// A compressed delta against a reference item.
    Delta(Enc::Output),
}

impl<I, Enc: Encoder<I, Dec> + ?Sized, Dec: Decoder<I, Enc> + ?Sized> CodecItem<I, Enc, Dec> {
    /// Create a new `CodecItem` from an uncompressed item.
    pub const fn new_uncompressed(item: I) -> Self {
        Self::Uncompressed(item)
    }

    /// Create a new `CodecItem` from a delta.
    pub const fn new_delta(delta: Enc::Output) -> Self {
        Self::Delta(delta)
    }

    /// Encode the item using the provided encoder and reference item.
    pub fn encode(&mut self, encoder: &Enc, reference: Option<&I>) {
        match self {
            Self::Uncompressed(item) => {
                let delta = reference.map_or_else(
                    || encoder.encode_root(item),
                    |reference| encoder.encode(item, reference),
                );
                *self = Self::Delta(delta);
            }
            Self::Delta(_) => {
                // Already encoded; do nothing.
            }
        }
    }

    /// Decode the item using the provided decoder and reference item.
    pub fn decode(&mut self, decoder: &Dec, reference: Option<&I>) {
        match self {
            Self::Uncompressed(_) => {
                // Already decoded; do nothing.
            }
            Self::Delta(delta) => {
                let item = reference.map_or_else(
                    || decoder.decode_root(delta),
                    |reference| decoder.decode(delta, reference),
                );
                *self = Self::Uncompressed(item);
            }
        }
    }
}
