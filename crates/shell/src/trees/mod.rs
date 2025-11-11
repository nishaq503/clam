//! Tree formats supported in the CLI.

use std::path::Path;

use abd_clam::{PartitionStrategy, Tree};

use crate::{
    data::ShellData,
    metrics::{Metric, cosine, euclidean, levenshtein},
};

pub enum VectorTree {
    F32(Tree<usize, Vec<f32>, f32, (), for<'a, 'b> fn(&'a Vec<f32>, &'b Vec<f32>) -> f32>),
    F64(Tree<usize, Vec<f64>, f64, (), for<'a, 'b> fn(&'a Vec<f64>, &'b Vec<f64>) -> f64>),
    I8(Tree<usize, Vec<i8>, i8, (), for<'a, 'b> fn(&'a Vec<i8>, &'b Vec<i8>) -> f32>),
    I16(Tree<usize, Vec<i16>, i16, (), for<'a, 'b> fn(&'a Vec<i16>, &'b Vec<i16>) -> f32>),
    I32(Tree<usize, Vec<i32>, i32, (), for<'a, 'b> fn(&'a Vec<i32>, &'b Vec<i32>) -> f32>),
    I64(Tree<usize, Vec<i64>, i64, (), for<'a, 'b> fn(&'a Vec<i64>, &'b Vec<i64>) -> f64>),
    U8(Tree<usize, Vec<u8>, u8, (), for<'a, 'b> fn(&'a Vec<u8>, &'b Vec<u8>) -> f32>),
    U16(Tree<usize, Vec<u16>, u16, (), for<'a, 'b> fn(&'a Vec<u16>, &'b Vec<u16>) -> f32>),
    U32(Tree<usize, Vec<u32>, u32, (), for<'a, 'b> fn(&'a Vec<u32>, &'b Vec<u32>) -> f32>),
    U64(Tree<usize, Vec<u64>, u64, (), for<'a, 'b> fn(&'a Vec<u64>, &'b Vec<u64>) -> f64>),
}

pub enum ShellTree {
    Levenshtein(Tree<String, String, u32, (), fn(&String, &String) -> u32>),
    Euclidean(VectorTree),
    Cosine(VectorTree),
}

impl ShellTree {
    /// Creates a new tree given a dataset and a metric.
    ///
    /// # Arguments
    ///
    /// - `inp_data`: The input data to build the tree from.
    /// - `metric`: The distance metric to use for the tree.
    /// - `seed`: The random seed to use.
    /// - `permuted`: Whether to apply depth-first-reordering to the data.
    ///
    /// # Returns
    ///
    /// A new tree and the transformed data.
    ///
    /// # Errors
    ///
    /// - If the dataset and metric are incompatible. The valid combinations
    ///   are:
    ///   - String data with Levenshtein metric.
    ///   - Float or Integer data with Euclidean or Cosine metrics.
    pub fn new(inp_data: ShellData, metric: &Metric) -> Result<ShellTree, String> {
        // TODO Najib: Implement a macro to handle the match arms more elegantly.
        match metric {
            Metric::Levenshtein => match inp_data {
                ShellData::String(items) => {
                    let strategy = PartitionStrategy::default();
                    let metric = levenshtein::<_, u32>;
                    let tree: Tree<_, _, _, (), _> = Tree::par_new(items, metric, &strategy, &|_, _, _| None)?;
                    Ok(ShellTree::Levenshtein(tree))
                }
                _ => Err("Levenshtein metric can only be used with string data".to_string()),
            },
            Metric::Euclidean => match inp_data {
                ShellData::String(_) => Err("Euclidean metric cannot be used for string data".to_string()),
                ShellData::F32(items) => {
                    let strategy = PartitionStrategy::default();
                    let items = items.into_iter().enumerate().collect::<Vec<_>>();
                    let tree = Tree::par_new(items, euclidean, &strategy, &|_, _, _| None)?;
                    Ok(ShellTree::Euclidean(VectorTree::F32(tree)))
                }
                _ => {
                    unimplemented!("Najib: Implement remaining match arms via macro");
                }
            },
            Metric::Cosine => match inp_data {
                ShellData::String(_) => Err("Cosine distance cannot be used for string data".to_string()),
                ShellData::F32(items) => {
                    let strategy = PartitionStrategy::default();
                    let items = items.into_iter().enumerate().collect::<Vec<_>>();
                    let tree = Tree::par_new(items, cosine, &strategy, &|_, _, _| None)?;
                    Ok(ShellTree::Cosine(VectorTree::F32(tree)))
                }
                _ => {
                    unimplemented!("Najib: Implement remaining match arms via macro");
                }
            },
        }
    }

    /// Saves the tree to the specified path using bincode.
    pub fn write_to<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        match self {
            Self::Levenshtein(tree) => {
                let bytes = tree.bitcode_encode().map_err(|e| e.to_string())?;
                std::fs::write(path, bytes).map_err(|e| e.to_string())
            },
            Self::Euclidean(tree) => tree.write_to(path),
            Self::Cosine(tree) => tree.write_to(path),
        }
    }

    /// Reads a tree from the specified path using bincode.
    pub fn read_from<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        match path.as_ref().extension().and_then(|s| s.to_str()) {
            Some("lev") => {
                let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
                let metric = levenshtein::<_, u32>;
                let tree = Tree::bitcode_decode(&bytes, metric).map_err(|e| e.to_string())?;
                Ok(ShellTree::Levenshtein(tree))
            }
            Some("euc") => Ok(ShellTree::Euclidean(VectorTree::read_from(path)?)),
            Some("cos") => Ok(ShellTree::Cosine(VectorTree::read_from(path)?)),
            _ => Err("Unsupported tree file extension".to_string()),
            
        }
    }
}

impl VectorTree {
    /// Saves the tree to the specified path using bitcode.
    pub fn write_to<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let (bytes, dtype) = match self {
            Self::F32(tree) => (tree.bitcode_encode(), "f4"),
            Self::F64(tree) => (tree.bitcode_encode(), "f8"),
            Self::I8(tree) => (tree.bitcode_encode(), "i1"),
            Self::I16(tree) => (tree.bitcode_encode(), "i2"),
            Self::I32(tree) => (tree.bitcode_encode(), "i4"),
            Self::I64(tree) => (tree.bitcode_encode(), "i8"),
            Self::U8(tree) => (tree.bitcode_encode(), "u1"),
            Self::U16(tree) => (tree.bitcode_encode(), "u2"),
            Self::U32(tree) => (tree.bitcode_encode(), "u4"),
            Self::U64(tree) => (tree.bitcode_encode(), "u8"),
        };
        let mut bytes = bytes.map_err(|e| e.to_string())?;
        bytes.extend_from_slice(dtype.as_bytes());

        std::fs::write(path, bytes).map_err(|e| e.to_string())
    }

    /// Reads a VectorTree from the specified path using bitcode.
    pub fn read_from<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        let dtype = std::str::from_utf8(&bytes[bytes.len() - 2..]).map_err(|e| e.to_string())?;
        let tree_bytes = &bytes[..bytes.len() - 2];
        let tree = match dtype {
            "f4" => {
                let metric = euclidean::<_, f32, f32>;
                let tree = Tree::bitcode_decode(tree_bytes, metric).map_err(|e| e.to_string())?;
                Self::F32(tree)
            }
            "f8" => {
                let metric = euclidean::<_, f64, f64>;
                let tree = Tree::bitcode_decode(tree_bytes, metric).map_err(|e| e.to_string())?;
                Self::F64(tree)
            }
            "i1" => {
                let metric = euclidean::<_, i8, f32>;
                let tree = Tree::bitcode_decode(tree_bytes, metric).map_err(|e| e.to_string())?;
                Self::I8(tree)
            }
            "i2" => {
                let metric = euclidean::<_, i16, f32>;
                let tree = Tree::bitcode_decode(tree_bytes, metric).map_err(|e| e.to_string())?;
                Self::I16(tree)
            }
            "i4" => {
                let metric = euclidean::<_, i32, f32>;
                let tree = Tree::bitcode_decode(tree_bytes, metric).map_err(|e| e.to_string())?;
                Self::I32(tree)
            }
            "i8" => {
                let metric = euclidean::<_, i64, f64>;
                let tree = Tree::bitcode_decode(tree_bytes, metric).map_err(|e| e.to_string())?;
                Self::I64(tree)
            }
            "u1" => {
                let metric = euclidean::<_, u8, f32>;
                let tree = Tree::bitcode_decode(tree_bytes, metric).map_err(|e| e.to_string())?;
                Self::U8(tree)
            }
            "u2" => {
                let metric = euclidean::<_, u16, f32>;
                let tree = Tree::bitcode_decode(tree_bytes, metric).map_err(|e| e.to_string())?;
                Self::U16(tree)
            }
            "u4" => {
                let metric = euclidean::<_, u32, f32>;
                let tree = Tree::bitcode_decode(tree_bytes, metric).map_err(|e| e.to_string())?;
                Self::U32(tree)
            }
            "u8" => {
                let metric = euclidean::<_, u64, f64>;
                let tree = Tree::bitcode_decode(tree_bytes, metric).map_err(|e| e.to_string())?;
                Self::U64(tree)
            }
            _ => return Err("Unsupported data type in tree file".to_string()),
        };

        Ok(tree)
    }
}

impl std::fmt::Display for VectorTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorTree::F32(_) => write!(f, "VectorTree<F32>"),
            VectorTree::F64(_) => write!(f, "VectorTree<F64>"),
            VectorTree::I8(_) => write!(f, "VectorTree<I8>"),
            VectorTree::I16(_) => write!(f, "VectorTree<I16>"),
            VectorTree::I32(_) => write!(f, "VectorTree<I32>"),
            VectorTree::I64(_) => write!(f, "VectorTree<I64>"),
            VectorTree::U8(_) => write!(f, "VectorTree<U8>"),
            VectorTree::U16(_) => write!(f, "VectorTree<U16>"),
            VectorTree::U32(_) => write!(f, "VectorTree<U32>"),
            VectorTree::U64(_) => write!(f, "VectorTree<U64>"),
        }
    }
}

impl std::fmt::Display for ShellTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShellTree::Levenshtein(_) => write!(f, "LevenshteinStringTree<U32>"),
            ShellTree::Euclidean(tree) => write!(f, "Euclidean{tree}"),
            ShellTree::Cosine(tree) => write!(f, "Cosine{tree}"),
        }
    }
}
