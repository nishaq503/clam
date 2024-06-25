//! Utilities for reading the CHAODA datasets.

use std::path::Path;

use distances::Number;
use ndarray::prelude::*;
use ndarray_npy::{ReadNpyExt, ReadableElement};

/// The result of reading a dataset.
///
/// The first element is the data and the second element is the labels.
pub type DataResult = (Vec<Vec<f32>>, Vec<bool>);

/// The datasets used for anomaly detection.
pub enum Data {
    Annthyroid,
    Arrhythmia,
    Backdoor,
    BreastW,
    Campaign,
    Cardio,
    CelebA,
    Census,
    Cover,
    Donors,
    Fraud,
    Glass,
    Http,
    Ionosphere,
    Lympho,
    Mammography,
    Mnist,
    Musk,
    OptDigits,
    PenDigits,
    Pima,
    Satellite,
    SatImage2,
    Shuttle,
    Smtp,
    Thyroid21,
    Thyroid,
    Vertebral,
    Vowels,
    Wbc,
    Wine,
}

impl Data {
    /// Read the dataset.
    ///
    /// # Arguments
    ///
    /// * `data_dir` - The directory containing the dataset.
    pub fn read(&self, data_dir: &Path) -> DataResult {
        match self {
            Self::Annthyroid => read_xy::<f64, u8>(data_dir, "annthyroid"),
            Self::Arrhythmia => read_xy::<f64, u8>(data_dir, "arrhythmia"),
            Self::Backdoor => read_xy::<f32, u8>(data_dir, "backdoor"),
            Self::BreastW => read_xy::<f64, u8>(data_dir, "breastw"),
            Self::Campaign => read_xy::<f32, u8>(data_dir, "campaign"),
            Self::Cardio => read_xy::<f64, u8>(data_dir, "cardio"),
            Self::CelebA => read_xy::<f32, u8>(data_dir, "celeba"),
            Self::Census => read_xy::<f32, u8>(data_dir, "census"),
            Self::Cover => read_xy::<f64, u8>(data_dir, "cover"),
            Self::Donors => read_xy::<f32, u8>(data_dir, "donors"),
            Self::Fraud => read_xy::<f32, u8>(data_dir, "fraud"),
            Self::Glass => read_xy::<f64, u8>(data_dir, "glass"),
            Self::Http => read_xy::<f64, u8>(data_dir, "http"),
            Self::Ionosphere => read_xy::<f64, u8>(data_dir, "ionosphere"),
            Self::Lympho => read_xy::<f64, u8>(data_dir, "lympho"),
            Self::Mammography => read_xy::<f64, u8>(data_dir, "mammography"),
            Self::Mnist => read_xy::<f64, u8>(data_dir, "mnist"),
            Self::Musk => read_xy::<f64, u8>(data_dir, "musk"),
            Self::OptDigits => read_xy::<f64, u8>(data_dir, "optdigits"),
            Self::PenDigits => read_xy::<f64, u8>(data_dir, "pendigits"),
            Self::Pima => read_xy::<f64, u8>(data_dir, "pima"),
            Self::Satellite => read_xy::<f64, u8>(data_dir, "satellite"),
            Self::SatImage2 => read_xy::<f64, u8>(data_dir, "satimage2"),
            Self::Shuttle => read_xy::<f64, u8>(data_dir, "shuttle"),
            Self::Smtp => read_xy::<f64, u8>(data_dir, "smtp"),
            Self::Thyroid21 => read_xy::<f32, u8>(data_dir, "thyroid21"),
            Self::Thyroid => read_xy::<f64, u8>(data_dir, "thyroid"),
            Self::Vertebral => read_xy::<f64, u8>(data_dir, "vertebral"),
            Self::Vowels => read_xy::<f64, u8>(data_dir, "vowels"),
            Self::Wbc => read_xy::<f64, u8>(data_dir, "wbc"),
            Self::Wine => read_xy::<f64, u8>(data_dir, "wine"),
        }
    }

    /// Read the training datasets from the paper
    pub fn read_paper_train(data_dir: &Path) -> Vec<(String, DataResult)> {
        vec![
            ("annthyroid".to_string(), Self::Annthyroid.read(data_dir)),
            ("mnist".to_string(), Self::Mnist.read(data_dir)),
            ("pendigits".to_string(), Self::PenDigits.read(data_dir)),
            ("satellite".to_string(), Self::Satellite.read(data_dir)),
            ("shuttle".to_string(), Self::Shuttle.read(data_dir)),
            ("thyroid".to_string(), Self::Thyroid.read(data_dir)),
        ]
    }

    /// Read the inference datasets from the paper
    pub fn read_paper_inference(data_dir: &Path) -> Vec<(String, DataResult)> {
        vec![
            ("arrhythmia".to_string(), Self::Arrhythmia.read(data_dir)),
            ("breastw".to_string(), Self::BreastW.read(data_dir)),
            ("cardio".to_string(), Self::Cardio.read(data_dir)),
            ("cover".to_string(), Self::Cover.read(data_dir)),
            ("glass".to_string(), Self::Glass.read(data_dir)),
            ("http".to_string(), Self::Http.read(data_dir)),
            ("ionosphere".to_string(), Self::Ionosphere.read(data_dir)),
            ("lympho".to_string(), Self::Lympho.read(data_dir)),
            ("mammography".to_string(), Self::Mammography.read(data_dir)),
            ("musk".to_string(), Self::Musk.read(data_dir)),
            ("optdigits".to_string(), Self::OptDigits.read(data_dir)),
            ("pima".to_string(), Self::Pima.read(data_dir)),
            ("satimage2".to_string(), Self::SatImage2.read(data_dir)),
            ("smtp".to_string(), Self::Smtp.read(data_dir)),
            ("vertebral".to_string(), Self::Vertebral.read(data_dir)),
            ("vowels".to_string(), Self::Vowels.read(data_dir)),
            ("wbc".to_string(), Self::Wbc.read(data_dir)),
            ("wine".to_string(), Self::Wine.read(data_dir)),
        ]
    }

    /// Read all the datasets
    #[allow(dead_code)]
    pub fn read_all(data_dir: &Path) -> Vec<(String, DataResult)> {
        let mut datasets = Self::read_paper_train(data_dir);
        datasets.extend(Self::read_paper_inference(data_dir));
        datasets.extend(vec![
            ("backdoor".to_string(), Self::Backdoor.read(data_dir)),
            ("campaign".to_string(), Self::Campaign.read(data_dir)),
            ("celeba".to_string(), Self::CelebA.read(data_dir)),
            ("census".to_string(), Self::Census.read(data_dir)),
            ("donors".to_string(), Self::Donors.read(data_dir)),
            ("fraud".to_string(), Self::Fraud.read(data_dir)),
            ("thyroid21".to_string(), Self::Thyroid21.read(data_dir)),
        ]);
        datasets
    }
}

fn read_xy<X, Y>(path: &Path, name: &str) -> DataResult
where
    X: Number + ReadableElement,
    Y: Number + ReadableElement,
{
    let x_path = path.join(format!("{}.npy", name));

    let reader = std::fs::File::open(x_path).unwrap();
    let x_data = Array2::<X>::read_npy(reader).unwrap();
    let x_data = x_data.mapv(|x| x.as_f32());
    let x_data = x_data
        .axis_iter(Axis(0))
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();

    let y_path = path.join(format!("{}_labels.npy", name));
    let reader = std::fs::File::open(y_path).unwrap();
    let y_data = Array1::<Y>::read_npy(reader).unwrap();
    let y_data = y_data.mapv(|y| y == Y::one()).to_vec();

    (x_data, y_data)
}
