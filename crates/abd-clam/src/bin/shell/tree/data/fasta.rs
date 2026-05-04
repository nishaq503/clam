//! Read and write FASTA datasets.

use seq_io::fasta::{self, Record};

/// Reads a FASTA file from the given path.
pub fn read<P: AsRef<std::path::Path>>(path: &P, remove_gaps: bool) -> Result<Vec<(String, String)>, Box<dyn std::error::Error + Send + Sync>> {
    let mut reader = fasta::Reader::from_path(path).map_err(|e| e.to_string())?;

    if remove_gaps {
        ftlog::info!("Removing gaps from sequences while reading FASTA file: {:?}", path.as_ref());
    }

    let mut records = Vec::new();

    while let Some(result) = reader.next() {
        let record = result.map_err(|e| e.to_string())?;

        let id = record.id().map_err(|e| e.to_string())?.to_string();
        let seq = record.full_seq();

        let seq = if remove_gaps {
            let gaps = [b'-', b'.']; // Common gap characters
            seq.iter().copied().filter(|b| !gaps.contains(b)).collect()
        } else {
            seq.to_vec()
        };
        let seq = String::from_utf8(seq).map_err(|e| e.to_string())?;
        records.push((id, seq));
    }

    Ok(records)
}

/// Writes an iterator of `AlignedSequence`s from Musals to a FASTA file at the given path.
pub fn write<P: AsRef<std::path::Path>>(path: &P, data: impl Iterator<Item = (String, String)>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let mut writer = std::io::BufWriter::new(file);

    for (id, seq) in data {
        let record = fasta::OwnedRecord {
            head: id.clone().into_bytes(),
            seq: seq.clone().into_bytes(),
        };
        record
            .write(&mut writer)
            .map_err(|e| e.to_string())
            .map_err(|e| format!("Error while writing record: {e}, id: {id}, seq: {seq}"))?;
    }

    Ok(())
}
