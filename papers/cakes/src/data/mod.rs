//! Utilities for the ANN-Benchmarks datasets.
//!
//! Their data can be found [here](https://github.com/erikbern/ann-benchmarks).

use std::path::{Path, PathBuf};

use rand::prelude::*;

/// Supported datasets from ANN-Benchmarks.
#[derive(Debug, PartialEq, Eq)]
#[allow(clippy::missing_docs_in_private_items, dead_code)]
pub enum AnnDataset {
    // Euclidean
    FashionMnist,
    Mnist,
    Sift,
    Gist,
    // Cosine
    Glove25,
    Glove50,
    Glove100,
    Glove200,
    DeepImage,
    NYTimes,
    LastFM,
}

impl AnnDataset {
    /// Returns the name of the dataset.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::FashionMnist => "FashionMnist",
            Self::Mnist => "Mnist",
            Self::Sift => "Sift",
            Self::Gist => "Gist",
            Self::Glove25 => "Glove25",
            Self::Glove50 => "Glove50",
            Self::Glove100 => "Glove100",
            Self::Glove200 => "Glove200",
            Self::DeepImage => "DeepImage",
            Self::NYTimes => "NYTimes",
            Self::LastFM => "LastFM",
        }
    }

    /// Returns the metric used by the dataset.
    #[allow(clippy::type_complexity)]
    pub const fn metric(&self) -> fn(&Vec<f32>, &Vec<f32>) -> f32 {
        match self {
            Self::FashionMnist | Self::Mnist | Self::Sift | Self::Gist => |x, y| distances::simd::euclidean_f32(x, y),
            _ => |x, y| distances::simd::cosine_f32(x, y),
        }
    }

    /// Returns all the supported datasets.
    pub fn all_datasets() -> Vec<Self> {
        vec![
            Self::FashionMnist,
            Self::Glove25,
            Self::Sift,
            Self::Glove100,
            // Self::NYTimes,
            Self::Mnist,
            Self::Glove50,
            Self::Glove200,
            Self::DeepImage,
            Self::LastFM,
            Self::Gist,
        ]
    }

    /// Returns the file name prefix for the dataset.
    pub const fn file_name_prefix(&self) -> &'static str {
        match self {
            Self::FashionMnist => "fashion-mnist",
            Self::Mnist => "mnist",
            Self::Sift => "sift",
            Self::Gist => "gist",
            Self::Glove25 => "glove-25",
            Self::Glove50 => "glove-50",
            Self::Glove100 => "glove-100",
            Self::Glove200 => "glove-200",
            Self::DeepImage => "deep-image",
            Self::NYTimes => "nytimes",
            Self::LastFM => "lastfm",
        }
    }

    /// Returns the path to the specified subset (train or test) of the dataset.
    fn subset_path<P: AsRef<Path>>(&self, base: &P, subset: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let path = base.as_ref().join(format!("{}-{}.npy", self.file_name_prefix(), subset.to_ascii_lowercase()));
        if path.exists() {
            Ok(path)
        } else {
            Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("{subset} subset not found: {path:?}"),
            )))
        }
    }

    /// Reads the specified subset (train or test) of the dataset.
    fn read_subset<P: AsRef<Path>, R: rand::Rng>(&self, base: &P, subset: &str, rng: Option<&mut R>) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        let path = self.subset_path(base, subset)?;
        let arr = ndarray_npy::read_npy::<_, ndarray::Array2<f32>>(path).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
        let mut vec = arr.outer_iter().map(|row| row.to_vec()).collect::<Vec<_>>();
        if let Some(rng) = rng {
            vec.shuffle(rng);
        }
        Ok(vec)
    }

    /// Reads the training subset of the dataset.
    pub fn read_train<P: AsRef<Path>, R: rand::Rng>(&self, base: &P, rng: Option<&mut R>) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        self.read_subset(base, "train", rng)
    }

    /// Reads the test subset of the dataset.
    pub fn read_test<P: AsRef<Path>, R: rand::Rng>(&self, base: &P, rng: Option<&mut R>) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        self.read_subset(base, "test", rng)
    }
}
