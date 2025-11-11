//! Data formats supported in the CLI.

use std::path::Path;
use std::str::FromStr;

mod fasta;
pub mod npy;

/// Reads the data from the file at the given path.
pub fn read<P: AsRef<Path>>(path: P) -> Result<ShellData, String> {
    match Format::from(&path) {
        Format::Npy => npy::NpyType::read(path),
        Format::Fasta => fasta::read(path).map(ShellData::String),
    }
}

/// Data formats supported in the CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Format {
    /// Npy array format.
    Npy,
    /// FASTA format.
    Fasta,
}

impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Format::Npy => write!(f, "npy"),
            Format::Fasta => write!(f, "fasta"),
        }
    }
}

impl<P: AsRef<Path>> From<P> for Format {
    fn from(path: P) -> Self {
        match path.as_ref().extension().and_then(|s| s.to_str()) {
            Some("npy") => Format::Npy,
            Some("fasta") => Format::Fasta,
            Some(ext) => panic!("Unknown data format {ext} for path: {}", path.as_ref().display()),
            None => panic!(
                "Could not determine data format without extension for path: {}",
                path.as_ref().display()
            ),
        }
    }
}

impl FromStr for Format {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "npy" => Ok(Format::Npy),
            "fasta" => Ok(Format::Fasta),
            _ => Err(format!("Unknown format: '{}'. Use 'npy' or 'fasta'.", s)),
        }
    }
}

/// Data types supported for generated datasets.
#[derive(Debug, Clone, Copy)]
pub enum DataType {
    /// 32-bit floating point
    F32,
    /// 64-bit floating point
    F64,
    /// 8-bit signed integer
    I8,
    /// 16-bit signed integer
    I16,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// 8-bit unsigned integer
    U8,
    /// 16-bit unsigned integer
    U16,
    /// 32-bit unsigned integer
    U32,
    /// 64-bit unsigned integer
    U64,
    /// String sequences
    String,
}

impl FromStr for DataType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "f32" => Ok(DataType::F32),
            "f64" => Ok(DataType::F64),
            "i8" => Ok(DataType::I8),
            "i16" => Ok(DataType::I16),
            "i32" => Ok(DataType::I32),
            "i64" => Ok(DataType::I64),
            "u8" => Ok(DataType::U8),
            "u16" => Ok(DataType::U16),
            "u32" => Ok(DataType::U32),
            "u64" => Ok(DataType::U64),
            "string" => Ok(DataType::String),
            _ => Err(format!("Unknown data type: {s}")),
        }
    }
}

#[derive(bitcode::Encode, bitcode::Decode, serde::Deserialize, serde::Serialize)]
pub enum ShellData {
    /// Vec of sequences and their metadata from FASTA files.
    String(Vec<(String, String)>),
    /// Vec of various numeric types from NPY files.
    F32(Vec<Vec<f32>>),
    F64(Vec<Vec<f64>>),
    I8(Vec<Vec<i8>>),
    I16(Vec<Vec<i16>>),
    I32(Vec<Vec<i32>>),
    I64(Vec<Vec<i64>>),
    U8(Vec<Vec<u8>>),
    U16(Vec<Vec<u16>>),
    U32(Vec<Vec<u32>>),
    U64(Vec<Vec<u64>>),
}

impl core::fmt::Display for ShellData {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ShellData::String(_) => write!(f, "String data"),
            ShellData::F32(_) => write!(f, "F32 data"),
            ShellData::F64(_) => write!(f, "F64 data"),
            ShellData::I8(_) => write!(f, "I8 data"),
            ShellData::I16(_) => write!(f, "I16 data"),
            ShellData::I32(_) => write!(f, "I32 data"),
            ShellData::I64(_) => write!(f, "I64 data"),
            ShellData::U8(_) => write!(f, "U8 data"),
            ShellData::U16(_) => write!(f, "U16 data"),
            ShellData::U32(_) => write!(f, "U32 data"),
            ShellData::U64(_) => write!(f, "U64 data"),
        }
    }
}

impl ShellData {
    /// Create a slice of a ShellData from start to end indices.
    pub fn slice(&self, start: usize, end: usize) -> Self {
        match self {
            Self::String(data) => Self::String(data[start..end].to_vec()),
            Self::F32(data) => Self::F32(data[start..end].to_vec()),
            Self::F64(data) => Self::F64(data[start..end].to_vec()),
            Self::I8(data) => Self::I8(data[start..end].to_vec()),
            Self::I16(data) => Self::I16(data[start..end].to_vec()),
            Self::I32(data) => Self::I32(data[start..end].to_vec()),
            Self::I64(data) => Self::I64(data[start..end].to_vec()),
            Self::U8(data) => Self::U8(data[start..end].to_vec()),
            Self::U16(data) => Self::U16(data[start..end].to_vec()),
            Self::U32(data) => Self::U32(data[start..end].to_vec()),
            Self::U64(data) => Self::U64(data[start..end].to_vec()),
        }
    }

    /// Writes the ShellData to a file at the given path.
    pub fn write<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        match self {
            Self::String(data) => fasta::write(path, data),
            Self::F32(data) => npy::write_npy(path, data),
            Self::F64(data) => npy::write_npy(path, data),
            Self::I8(data) => npy::write_npy(path, data),
            Self::I16(data) => npy::write_npy(path, data),
            Self::I32(data) => npy::write_npy(path, data),
            Self::I64(data) => npy::write_npy(path, data),
            Self::U8(data) => npy::write_npy(path, data),
            Self::U16(data) => npy::write_npy(path, data),
            Self::U32(data) => npy::write_npy(path, data),
            Self::U64(data) => npy::write_npy(path, data),
        }
    }
}
