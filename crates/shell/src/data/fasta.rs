use std::path::Path;

/// Reads a FASTA file from the given path.
#[allow(dead_code, unused_variables)]
pub fn read<P: AsRef<Path>>(path: P) -> Result<Vec<(String, String)>, String> {
    todo!("Najib: Implement reading FASTA files")
}

/// Writes a FASTA file to the given path.
pub fn write<P: AsRef<Path>>(path: P, data: &[(String, String)]) -> Result<(), String> {
    use std::io::Write;

    let file = std::fs::File::create(path).map_err(|e| format!("Failed to create FASTA file: {e}"))?;
    let mut writer = std::io::BufWriter::new(file);

    // Write each sequence with a simple numeric ID
    for (id, sequence) in data {
        writeln!(writer, ">{id}").map_err(|e| format!("Failed to write sequence header: {e}"))?;
        writeln!(writer, "{sequence}").map_err(|e| format!("Failed to write sequence data: {e}"))?;
    }

    writer.flush().map_err(|e| format!("Failed to flush FASTA file: {e}"))?;

    Ok(())
}
