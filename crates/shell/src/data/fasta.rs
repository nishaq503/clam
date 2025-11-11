use std::path::Path;

use bio::io::fasta;

/// Reads a FASTA file from the given path.
pub fn read<P: AsRef<Path> + core::fmt::Debug>(path: P) -> Result<Vec<(String, String)>, String> {
    let reader = fasta::Reader::from_file(&path).map_err(|e| e.to_string())?;

    let mut records = Vec::new();
    for result in reader.records() {
        let record = result.map_err(|e| e.to_string())?;
        let id = record.id().to_string();
        let seq = String::from_utf8(record.seq().to_vec()).map_err(|e| e.to_string())?;
        records.push((id, seq));
    }

    Ok(records)
}

/// Writes a FASTA file to the given path.
pub fn write<P: AsRef<Path>>(path: P, data: &[(String, String)]) -> Result<(), String> {
    let mut writer = fasta::Writer::to_file(&path).map_err(|e| e.to_string())?;

    for (id, seq) in data {
        let record = fasta::Record::with_attrs(id, None, seq.as_bytes());
        writer
            .write_record(&record)
            .map_err(|e| e.to_string())
            .map_err(|e| format!("Error while writing record: {e}, id: {id}, seq: {seq}"))?;
    }

    Ok(())
}
