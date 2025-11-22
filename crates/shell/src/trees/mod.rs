//! Tree formats supported in the CLI.

use std::path::{Path, PathBuf};

use abd_clam::{DistanceValue, PartitionStrategy, Tree, cakes::Search};

use crate::{
    commands::cakes::{AlgorithmResult, QueryResult, SearchResults},
    data::{MusalsSequence, OutputFormat, ShellData},
    metrics::{Metric, cosine, euclidean},
    search::ShellCakes,
};

macro_rules! tree_type {
    ($id:ty, $i:ty, $t:ty) => {
        Tree<$id, $i, $t, (), fn(&$i, &$i) -> $t>
    };
}

pub enum VectorTree {
    F32(tree_type!(usize, Vec<f32>, f32)),
    F64(tree_type!(usize, Vec<f64>, f64)),
    I8(tree_type!(usize, Vec<i8>, f32)),
    I16(tree_type!(usize, Vec<i16>, f32)),
    I32(tree_type!(usize, Vec<i32>, f32)),
    I64(tree_type!(usize, Vec<i64>, f64)),
    U8(tree_type!(usize, Vec<u8>, f32)),
    U16(tree_type!(usize, Vec<u16>, f32)),
    U32(tree_type!(usize, Vec<u32>, f32)),
    U64(tree_type!(usize, Vec<u64>, f64)),
}

#[expect(clippy::type_complexity)]
pub enum ShellTree {
    Levenshtein(
        Tree<String, MusalsSequence, u32, (), Box<dyn Fn(&MusalsSequence, &MusalsSequence) -> u32 + Send + Sync>>,
    ),
    Euclidean(VectorTree),
    Cosine(VectorTree),
}

macro_rules! st_new_vector_arm {
    ($items:ident, $metric:ident, $ty:ty, $st_var:ident, $vt_var:ident) => {{
        let strategy = PartitionStrategy::default();
        let items = $items.into_iter().enumerate().collect::<Vec<_>>();
        let metric: fn(&_, &_) -> $ty = $metric::<_, _, _>;
        let tree = Tree::par_new(items, metric, &strategy, &|_| None)?;
        Ok(ShellTree::$st_var(VectorTree::$vt_var(tree)))
    }};
}

impl ShellTree {
    /// Creates a new tree given a dataset and a metric.
    ///
    /// # Arguments
    ///
    /// - `inp_data`: The input data to build the tree from.
    /// - `metric`: The distance metric to use for the tree.
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
        match metric {
            Metric::Levenshtein => match inp_data {
                ShellData::String(items) => {
                    let sz_device = stringzilla::szs::DeviceScope::default().map_err(|e| e.to_string())?;
                    let sz_engine = stringzilla::szs::LevenshteinDistances::new(&sz_device, 0, 1, 1, 1)
                        .map_err(|e| e.to_string())?;
                    let metric = move |a: &MusalsSequence, b: &MusalsSequence| -> u32 {
                        let distances = sz_engine.compute(&sz_device, &[a.as_ref()], &[b.as_ref()]).unwrap();
                        distances[0] as u32
                    };
                    let metric = Box::new(metric) as Box<dyn Fn(&MusalsSequence, &MusalsSequence) -> u32 + Send + Sync>;

                    let strategy = PartitionStrategy::default();
                    let tree = Tree::par_new(items, metric, &strategy, &|_| None)?;
                    Ok(ShellTree::Levenshtein(tree))
                }
                _ => Err("Levenshtein metric can only be used with string data".to_string()),
            },
            Metric::Euclidean => match inp_data {
                ShellData::String(_) => Err("Euclidean metric cannot be used for string data".to_string()),
                ShellData::F32(items) => st_new_vector_arm!(items, euclidean, f32, Euclidean, F32),
                ShellData::F64(items) => st_new_vector_arm!(items, euclidean, f64, Euclidean, F64),
                ShellData::I8(items) => st_new_vector_arm!(items, euclidean, f32, Euclidean, I8),
                ShellData::I16(items) => st_new_vector_arm!(items, euclidean, f32, Euclidean, I16),
                ShellData::I32(items) => st_new_vector_arm!(items, euclidean, f32, Euclidean, I32),
                ShellData::I64(items) => st_new_vector_arm!(items, euclidean, f64, Euclidean, I64),
                ShellData::U8(items) => st_new_vector_arm!(items, euclidean, f32, Euclidean, U8),
                ShellData::U16(items) => st_new_vector_arm!(items, euclidean, f32, Euclidean, U16),
                ShellData::U32(items) => st_new_vector_arm!(items, euclidean, f32, Euclidean, U32),
                ShellData::U64(items) => st_new_vector_arm!(items, euclidean, f64, Euclidean, U64),
            },
            Metric::Cosine => match inp_data {
                ShellData::String(_) => Err("Cosine distance cannot be used for string data".to_string()),
                ShellData::F32(items) => st_new_vector_arm!(items, cosine, f32, Cosine, F32),
                ShellData::F64(items) => st_new_vector_arm!(items, cosine, f64, Cosine, F64),
                ShellData::I8(items) => st_new_vector_arm!(items, cosine, f32, Cosine, I8),
                ShellData::I16(items) => st_new_vector_arm!(items, cosine, f32, Cosine, I16),
                ShellData::I32(items) => st_new_vector_arm!(items, cosine, f32, Cosine, I32),
                ShellData::I64(items) => st_new_vector_arm!(items, cosine, f64, Cosine, I64),
                ShellData::U8(items) => st_new_vector_arm!(items, cosine, f32, Cosine, U8),
                ShellData::U16(items) => st_new_vector_arm!(items, cosine, f32, Cosine, U16),
                ShellData::U32(items) => st_new_vector_arm!(items, cosine, f32, Cosine, U32),
                ShellData::U64(items) => st_new_vector_arm!(items, cosine, f64, Cosine, U64),
            },
        }
    }

    /// Search the tree with the given queries and algorithms.
    pub fn search<P: AsRef<Path> + core::fmt::Debug>(
        &self,
        queries: ShellData,
        algorithms: &[ShellCakes],
        out_path: P,
    ) -> Result<(), String> {
        match self {
            Self::Levenshtein(tree) => match queries {
                ShellData::String(queries) => {
                    let queries = queries.iter().map(|(_, seq)| seq.clone()).collect::<Vec<_>>();
                    search(tree, &queries, algorithms, out_path)
                }
                _ => Err("Levenshtein tree can only be searched with string queries".to_string()),
            },
            Self::Euclidean(tree) => match queries {
                ShellData::String(_) => Err("Euclidean tree cannot be searched with string queries".to_string()),
                _ => tree.search(queries, algorithms, out_path),
            },
            Self::Cosine(tree) => match queries {
                ShellData::String(_) => Err("Cosine tree cannot be searched with string queries".to_string()),
                _ => tree.search(queries, algorithms, out_path),
            },
        }
    }

    /// Saves the tree to the specified path using bincode.
    pub fn write_to<P: AsRef<Path>>(&self, out_dir: P, suffix: Option<&str>) -> Result<(), String> {
        match self {
            Self::Levenshtein(tree) => {
                let suffix = suffix.map_or_else(|| "lev".to_string(), |s| format!("{s}-lev"));
                let path = out_dir.as_ref().join(format!("tree-{suffix}.bin"));
                let bytes = tree.bitcode_encode().map_err(|e| e.to_string())?;
                std::fs::write(path, bytes).map_err(|e| e.to_string())
            }
            Self::Euclidean(tree) => {
                let suffix = suffix.map_or_else(|| "euc".to_string(), |s| format!("{s}-euc"));
                let path = out_dir.as_ref().join(format!("tree-{suffix}.bin"));
                tree.write_to(path)
            }
            Self::Cosine(tree) => {
                let suffix = suffix.map_or_else(|| "cos".to_string(), |s| format!("{s}-cos"));
                let path = out_dir.as_ref().join(format!("tree-{suffix}.bin"));
                tree.write_to(path)
            }
        }
    }

    /// Reads a tree from the specified input directory using bincode.
    pub fn read_from<P: AsRef<Path>>(inp_dir: P) -> Result<(Self, PathBuf), String> {
        let inp_dir = inp_dir.as_ref();
        // Find a ".bin" file whose name starts with "tree-".
        let tree_path = std::fs::read_dir(inp_dir)
            .map_err(|e| format!("Failed to read input directory {}: {}", inp_dir.display(), e))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .find(|path| {
                path.is_file()
                    && path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.starts_with("tree-") && name.ends_with(".bin"))
            })
            .ok_or_else(|| format!("No tree file found in directory {}", inp_dir.display()))?;

        // The metric name is the last three characters before the ".bin" extension.
        let metric_name = tree_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .and_then(|stem| stem.split('-').next_back())
            .ok_or_else(|| "Failed to determine metric from tree file name".to_string())?;

        let tree = match metric_name {
            "lev" => {
                let sz_device = stringzilla::szs::DeviceScope::default().map_err(|e| e.to_string())?;
                let sz_engine =
                    stringzilla::szs::LevenshteinDistances::new(&sz_device, 0, 1, 1, 1).map_err(|e| e.to_string())?;
                let metric = move |a: &MusalsSequence, b: &MusalsSequence| -> u32 {
                    let distances = sz_engine.compute(&sz_device, &[a.as_ref()], &[b.as_ref()]).unwrap();
                    distances[0] as u32
                };
                let metric = Box::new(metric) as Box<dyn Fn(&MusalsSequence, &MusalsSequence) -> u32 + Send + Sync>;

                let bytes = std::fs::read(&tree_path).map_err(|e| e.to_string())?;
                let tree = Tree::bitcode_decode(&bytes, metric).map_err(|e| e.to_string())?;
                Ok(ShellTree::Levenshtein(tree))
            }
            "euc" => Ok(ShellTree::Euclidean(VectorTree::read_from(&tree_path)?)),
            "cos" => Ok(ShellTree::Cosine(VectorTree::read_from(&tree_path)?)),
            _ => Err("Unsupported metric in tree file".to_string()),
        };

        tree.map(|t| (t, tree_path))
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
                let metric: fn(&_, &_) -> f32 = euclidean::<_, f32, f32>;
                let tree = Tree::bitcode_decode(tree_bytes, metric).map_err(|e| e.to_string())?;
                Self::F32(tree)
            }
            "f8" => {
                let metric: fn(&_, &_) -> f64 = euclidean::<_, f64, f64>;
                let tree = Tree::bitcode_decode(tree_bytes, metric).map_err(|e| e.to_string())?;
                Self::F64(tree)
            }
            "i1" => {
                let metric: fn(&_, &_) -> f32 = euclidean::<_, i8, f32>;
                let tree = Tree::bitcode_decode(tree_bytes, metric).map_err(|e| e.to_string())?;
                Self::I8(tree)
            }
            "i2" => {
                let metric: fn(&_, &_) -> f32 = euclidean::<_, i16, f32>;
                let tree = Tree::bitcode_decode(tree_bytes, metric).map_err(|e| e.to_string())?;
                Self::I16(tree)
            }
            "i4" => {
                let metric: fn(&_, &_) -> f32 = euclidean::<_, i32, f32>;
                let tree = Tree::bitcode_decode(tree_bytes, metric).map_err(|e| e.to_string())?;
                Self::I32(tree)
            }
            "i8" => {
                let metric: fn(&_, &_) -> f64 = euclidean::<_, i64, f64>;
                let tree = Tree::bitcode_decode(tree_bytes, metric).map_err(|e| e.to_string())?;
                Self::I64(tree)
            }
            "u1" => {
                let metric: fn(&_, &_) -> f32 = euclidean::<_, u8, f32>;
                let tree = Tree::bitcode_decode(tree_bytes, metric).map_err(|e| e.to_string())?;
                Self::U8(tree)
            }
            "u2" => {
                let metric: fn(&_, &_) -> f32 = euclidean::<_, u16, f32>;
                let tree = Tree::bitcode_decode(tree_bytes, metric).map_err(|e| e.to_string())?;
                Self::U16(tree)
            }
            "u4" => {
                let metric: fn(&_, &_) -> f32 = euclidean::<_, u32, f32>;
                let tree = Tree::bitcode_decode(tree_bytes, metric).map_err(|e| e.to_string())?;
                Self::U32(tree)
            }
            "u8" => {
                let metric: fn(&_, &_) -> f64 = euclidean::<_, u64, f64>;
                let tree = Tree::bitcode_decode(tree_bytes, metric).map_err(|e| e.to_string())?;
                Self::U64(tree)
            }
            _ => return Err("Unsupported data type in tree file".to_string()),
        };

        Ok(tree)
    }

    /// Search the tree with the given queries and algorithms.
    pub fn search<P: AsRef<Path> + core::fmt::Debug>(
        &self,
        queries: ShellData,
        algorithms: &[ShellCakes],
        out_path: P,
    ) -> Result<(), String> {
        match (self, queries) {
            (Self::F32(tree), ShellData::F32(queries)) => search(tree, &queries, algorithms, out_path),
            (Self::F64(tree), ShellData::F64(queries)) => search(tree, &queries, algorithms, out_path),
            (Self::I8(tree), ShellData::I8(queries)) => search(tree, &queries, algorithms, out_path),
            (Self::I16(tree), ShellData::I16(queries)) => search(tree, &queries, algorithms, out_path),
            (Self::I32(tree), ShellData::I32(queries)) => search(tree, &queries, algorithms, out_path),
            (Self::I64(tree), ShellData::I64(queries)) => search(tree, &queries, algorithms, out_path),
            (Self::U8(tree), ShellData::U8(queries)) => search(tree, &queries, algorithms, out_path),
            (Self::U16(tree), ShellData::U16(queries)) => search(tree, &queries, algorithms, out_path),
            (Self::U32(tree), ShellData::U32(queries)) => search(tree, &queries, algorithms, out_path),
            (Self::U64(tree), ShellData::U64(queries)) => search(tree, &queries, algorithms, out_path),
            _ => Err("Query data type does not match vectors type in tree".to_string()),
        }
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

fn search<Id, I, T, A, M, P>(
    tree: &Tree<Id, I, T, A, M>,
    queries: &[I],
    algs: &[ShellCakes],
    out_path: P,
) -> Result<(), String>
where
    T: DistanceValue + 'static,
    M: Fn(&I, &I) -> T,
    P: AsRef<Path> + core::fmt::Debug,
    <T as core::str::FromStr>::Err: std::fmt::Display,
{
    let mut all_results = SearchResults { results: Vec::new() };

    for (i, query) in queries.iter().enumerate() {
        println!("Processing query {i}");

        let mut query_result = QueryResult {
            query_index: i,
            algorithms: Vec::new(),
        };

        for alg in algs {
            let alg = alg.get()?;
            let result = alg.search(tree, query);
            println!("Result {}: {result:?}", alg.name());

            // Convert result to f64 for serialization consistency
            let neighbors: Vec<(usize, f64)> = result
                .into_iter()
                .map(|(idx, dist)| (idx, dist.to_f64().unwrap()))
                .collect();

            query_result.algorithms.push(AlgorithmResult {
                algorithm: alg.name(),
                neighbors,
            });
        }

        all_results.results.push(query_result);
    }

    OutputFormat::write(out_path, &all_results)
}
