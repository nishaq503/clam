//! Utilities for reading the dataset.
//!
//! The main paper of comparison is [here](https://arxiv.org/pdf/2102.01956&ved=2ahUKEwj04Jvy0eD8AhULk2oFHYFBCeMQFnoECAwQAQ&usg=AOvVaw2vo7nW3FMB6VPJsA1MXyvc)

use std::path::Path;

use abd_clam::Instance;
use distances::Number;
use ndarray::prelude::*;
use num_complex::Complex32;
use rand::prelude::*;
use rayon::prelude::*;

use super::metrics::C32;

/// An SNR level.
pub type Snr = i32;

/// All possible modulation modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModulationMode {
    /// On-Off Keying
    Ook,
    /// 4 Amplitude Shift Keying (4 Symbols in the constellation)
    Ask4,
    /// 8 Amplitude Shift Keying
    Ask8,
    /// Binary Phase Shift Keying
    Bpsk,
    /// Quadrature Phase Shift Keying
    Qpsk,
    /// 8 Phase Shift Keying
    Psk8,
    /// 16 Phase Shift Keying
    Psk16,
    /// 32 Phase Shift Keying
    Psk32,
    /// 16 Amplitude and Phase Shift Keying
    Apsk16,
    /// 32 Amplitude and Phase Shift Keying
    Apsk32,
    /// 64 Amplitude and Phase Shift Keying
    Apsk64,
    /// 128 Amplitude and Phase Shift Keying
    Apsk128,
    /// 16 Quadrature Amplitude Modulation
    Qam16,
    /// 32 Quadrature Amplitude Modulation
    Qam32,
    /// 64 Quadrature Amplitude Modulation
    Qam64,
    /// 128 Quadrature Amplitude Modulation
    Qam128,
    /// 256 Quadrature Amplitude Modulation
    Qam256,
    /// AM Single Side Band With Carrier
    AmSsbWc,
    /// AM Single Side Band Suppressed Carrier
    AmSsbSc,
    /// AM Double Side Band With Carrier
    AmDsbWc,
    /// AM Double Side Band Suppressed Carrier
    AmDsbSc,
    /// Frequency Modulation
    Fm,
    /// Gaussian Minimum Shift Keying
    Gmsk,
    /// Offset Quadrature Phase Shift Keying
    Oqpsk,
}

impl ModulationMode {
    /// Returns the name of the modulation mode.
    #[allow(dead_code)]
    const fn name(&self) -> &str {
        match self {
            Self::Apsk128 => "128APSK",
            Self::Qam128 => "128QAM",
            Self::Apsk16 => "16APSK",
            Self::Psk16 => "16PSK",
            Self::Qam16 => "16QAM",
            Self::Qam256 => "256QAM",
            Self::Apsk32 => "32APSK",
            Self::Psk32 => "32PSK",
            Self::Qam32 => "32QAM",
            Self::Ask4 => "4ASK",
            Self::Apsk64 => "64APSK",
            Self::Qam64 => "64QAM",
            Self::Ask8 => "8ASK",
            Self::Psk8 => "8PSK",
            Self::AmDsbSc => "AM-DSB-SC",
            Self::AmDsbWc => "AM-DSB-WC",
            Self::AmSsbSc => "AM-SSB-SC",
            Self::AmSsbWc => "AM-SSB-WC",
            Self::Bpsk => "BPSK",
            Self::Fm => "FM",
            Self::Gmsk => "GMSK",
            Self::Ook => "OOK",
            Self::Oqpsk => "OQPSK",
            Self::Qpsk => "QPSK",
        }
    }

    /// Returns the name of the data file for this modulation mode.
    const fn data_name(&self) -> &str {
        match self {
            Self::Apsk128 => "mod_128APSK.h5",
            Self::Qam128 => "mod_128QAM.h5",
            Self::Apsk16 => "mod_16APSK.h5",
            Self::Psk16 => "mod_16PSK.h5",
            Self::Qam16 => "mod_16QAM.h5",
            Self::Qam256 => "mod_256QAM.h5",
            Self::Apsk32 => "mod_32APSK.h5",
            Self::Psk32 => "mod_32PSK.h5",
            Self::Qam32 => "mod_32QAM.h5",
            Self::Ask4 => "mod_4ASK.h5",
            Self::Apsk64 => "mod_64APSK.h5",
            Self::Qam64 => "mod_64QAM.h5",
            Self::Ask8 => "mod_8ASK.h5",
            Self::Psk8 => "mod_8PSK.h5",
            Self::AmDsbSc => "mod_AM-DSB-SC.h5",
            Self::AmDsbWc => "mod_AM-DSB-WC.h5",
            Self::AmSsbSc => "mod_AM-SSB-SC.h5",
            Self::AmSsbWc => "mod_AM-SSB-WC.h5",
            Self::Bpsk => "mod_BPSK.h5",
            Self::Fm => "mod_FM.h5",
            Self::Gmsk => "mod_GMSK.h5",
            Self::Ook => "mod_OOK.h5",
            Self::Oqpsk => "mod_OQPSK.h5",
            Self::Qpsk => "mod_QPSK.h5",
        }
    }

    /// Returns the number of possible modulation modes.
    pub const fn variant_count() -> usize {
        24
    }

    /// Returns all possible modulation modes.
    const fn variants() -> [Self; 24] {
        [
            Self::Apsk128,
            Self::Qam128,
            Self::Apsk16,
            Self::Psk16,
            Self::Qam16,
            Self::Qam256,
            Self::Apsk32,
            Self::Psk32,
            Self::Qam32,
            Self::Ask4,
            Self::Apsk64,
            Self::Qam64,
            Self::Ask8,
            Self::Psk8,
            Self::AmDsbSc,
            Self::AmDsbWc,
            Self::AmSsbSc,
            Self::AmSsbWc,
            Self::Bpsk,
            Self::Fm,
            Self::Gmsk,
            Self::Ook,
            Self::Oqpsk,
            Self::Qpsk,
        ]
    }

    /// Returns a unique integer for each modulation mode.
    const fn to_index(self) -> u16 {
        match self {
            Self::Apsk128 => 0,
            Self::Qam128 => 1,
            Self::Apsk16 => 2,
            Self::Psk16 => 3,
            Self::Qam16 => 4,
            Self::Qam256 => 5,
            Self::Apsk32 => 6,
            Self::Psk32 => 7,
            Self::Qam32 => 8,
            Self::Ask4 => 9,
            Self::Apsk64 => 10,
            Self::Qam64 => 11,
            Self::Ask8 => 12,
            Self::Psk8 => 13,
            Self::AmDsbSc => 14,
            Self::AmDsbWc => 15,
            Self::AmSsbSc => 16,
            Self::AmSsbWc => 17,
            Self::Bpsk => 18,
            Self::Fm => 19,
            Self::Gmsk => 20,
            Self::Ook => 21,
            Self::Oqpsk => 22,
            Self::Qpsk => 23,
        }
    }

    /// Converts a unique integer to a modulation mode.
    fn from_index(i: u16) -> Result<Self, String> {
        match i {
            0 => Ok(Self::Apsk128),
            1 => Ok(Self::Qam128),
            2 => Ok(Self::Apsk16),
            3 => Ok(Self::Psk16),
            4 => Ok(Self::Qam16),
            5 => Ok(Self::Qam256),
            6 => Ok(Self::Apsk32),
            7 => Ok(Self::Psk32),
            8 => Ok(Self::Qam32),
            9 => Ok(Self::Ask4),
            10 => Ok(Self::Apsk64),
            11 => Ok(Self::Qam64),
            12 => Ok(Self::Ask8),
            13 => Ok(Self::Psk8),
            14 => Ok(Self::AmDsbSc),
            15 => Ok(Self::AmDsbWc),
            16 => Ok(Self::AmSsbSc),
            17 => Ok(Self::AmSsbWc),
            18 => Ok(Self::Bpsk),
            19 => Ok(Self::Fm),
            20 => Ok(Self::Gmsk),
            21 => Ok(Self::Ook),
            22 => Ok(Self::Oqpsk),
            23 => Ok(Self::Qpsk),
            _ => Err(format!("Unknown modulation mode index {i}.")),
        }
    }

    /// Converts the modulation mode to a byte array.
    pub fn to_bytes(self) -> Vec<u8> {
        self.to_index().to_be_bytes().to_vec()
    }

    /// Converts a byte array to a modulation mode.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut index_bytes = [0; 2];
        index_bytes.copy_from_slice(&bytes[..2]);
        let index = u16::from_be_bytes(index_bytes);
        Self::from_index(index)
    }
}

/// Samples at a single SNR level for a single modulation mode from the `RadioML` dataset.
struct Samples {
    /// -50dB for noise, otherwise one of (-20..=30).step_by(2)
    snr: Snr,

    /// the IQ data for building the tree
    train: Vec<Vec<C32>>,

    /// the IQ data for the queries
    queries: Vec<Vec<C32>>,
}

impl Samples {
    /// Creates a new `SnRLevel` from the given SNR level and IQ data.
    ///
    /// # Arguments
    ///
    /// * `snr` - The SNR level.
    /// * `iq` - The IQ data.
    /// * `sample_size` - The number of samples to use for training.
    /// * `num_queries` - The number of samples to use for queries.
    /// * `seed` - Optional seed for the random number generator for shuffling
    /// the data.
    ///
    /// # Errors
    ///
    /// * If the shape of the IQ data is not `[4096, 1024, 2]`.
    fn new(
        snr: Snr,
        iq: &Array3<f64>,
        sample_size: usize,
        num_queries: usize,
        seed: Option<u64>,
    ) -> Result<Self, String> {
        if iq.shape() == [4096, 1024, 2] {
            let mut samples = iq
                .axis_iter(Axis(0))
                .map(|sample| {
                    sample
                        .axis_iter(Axis(0))
                        .map(|row| (row[0], row[1]))
                        .map(|(i, q)| Complex32::from_polar(i.as_f32(), q.as_f32()))
                        .map(C32::from_complex)
                        .collect::<Vec<_>>()
                })
                .map(C32::normalize)
                .collect::<Vec<_>>();

            let mut rng = seed.map_or_else(StdRng::from_entropy, StdRng::seed_from_u64);
            samples.shuffle(&mut rng);
            samples.truncate(sample_size + num_queries);

            let (train_samples, query_samples) = samples.split_at(sample_size);
            let train_samples = train_samples.to_vec();
            let query_samples = query_samples.to_vec();

            Ok(Self {
                snr,
                train: train_samples,
                queries: query_samples,
            })
        } else {
            Err(format!(
                "Expected shape [4096, 1024, 2] but got {:?}.",
                iq.shape()
            ))
        }
    }

    /// Getter for the SNR level.
    const fn snr(&self) -> Snr {
        self.snr
    }

    /// Getter for the IQ data for building the tree.
    fn train_samples(&self) -> Vec<Vec<C32>> {
        self.train.clone()
    }

    /// Getter for the IQ data for the queries.
    fn query_samples(&self) -> Vec<Vec<C32>> {
        self.queries.clone()
    }
}

/// Contains the data for all SNR levels for a single modulation mode.
pub struct ModulationLevels {
    /// The IQ data for each SNR level.
    levels: Vec<Samples>,
}

impl ModulationLevels {
    /// Reads the IQ data for all SNR levels for the given modulation mode from
    /// the given HDF5 file.
    ///
    /// # Arguments
    ///
    /// * `handle` - The HDF5 file handle.
    /// * `modulation` - The modulation mode.
    /// * `sample_size` - The number of samples to use for training.
    /// * `num_queries` - The number of samples to use for queries.
    /// * `seed` - Optional seed for the random number generator for shuffling
    /// the data.
    ///
    /// # Errors
    ///
    /// * If the IQ data could not be read from the HDF5 file.
    /// * If the IQ data has the wrong shape.
    /// * If the number of SNR levels is not 26.
    fn new(
        handle: &hdf5::File,
        sample_size: usize,
        num_queries: usize,
        seed: Option<u64>,
    ) -> Result<Self, String> {
        let mut all_iq: Array3<f64> = handle
            .dataset("X")
            .map_err(|reason| format!("Could not read `X` because {reason}"))?
            .read()
            .map_err(|e| e.to_string())?;

        assert_eq!(
            all_iq.shape(),
            [26 * 4096, 1024, 2],
            "iq data had the wrong shape"
        );

        let mut all_iq = all_iq.view_mut();
        let mut levels = Vec::new();
        for snr in (-20..=30).step_by(2) {
            let (iq, rest) = all_iq.split_at(Axis(0), 4096);
            all_iq = rest;
            let iq = iq.to_owned();
            levels.push(Samples::new(snr, &iq, sample_size, num_queries, seed)?);
        }

        if levels.len() == 26 {
            Ok(Self { levels })
        } else {
            Err(format!("Expected 26 levels but got {}.", levels.len()))
        }
    }
}

/// Reads the IQ data for all modulation modes from the given directory.
///
/// # Arguments
///
/// * `input_dir` - The directory containing the HDF5 files.
/// * `sample_size` - The number of samples to use for training.
/// * `num_queries` - The number of samples to use for queries.
/// * `seed` - Optional seed for the random number generator for shuffling
/// the data.
///
/// # Returns
///
/// * The IQ data for all modulation modes and SNR levels.
///
/// # Errors
///
/// * If the IQ data could not be read from the HDF5 files.
#[allow(clippy::type_complexity)]
pub fn read(
    input_dir: &Path,
    snr: Snr,
    sample_size: usize,
    num_queries: usize,
    seed: Option<u64>,
) -> Result<[Vec<(RadioMLMetadata, Vec<C32>)>; 2], String> {
    let data = ModulationMode::variants()
        .into_par_iter()
        .map(|modulation| {
            let file_path = input_dir.join(modulation.data_name());
            let handle = hdf5::File::open(file_path).map_err(|e| e.to_string())?;
            let mode = ModulationLevels::new(&handle, sample_size, num_queries, seed)?;

            let level = mode
                .levels
                .into_iter()
                .find(|level| level.snr() == snr)
                .ok_or_else(|| format!("Could not find SNR level {snr}."))?;

            let query_samples = level
                .query_samples()
                .into_iter()
                .map(|sample| (RadioMLMetadata { modulation, snr }, sample))
                .collect::<Vec<_>>();

            let train_samples = level
                .train_samples()
                .into_iter()
                .map(|sample| (RadioMLMetadata { modulation, snr }, sample))
                .collect::<Vec<_>>();

            Ok((train_samples, query_samples))
        })
        .collect::<Result<Vec<_>, String>>()?;

    let (train_data, query_data) = data.into_iter().fold(
        (Vec::new(), Vec::new()),
        |(mut train_data, mut query_data), (train_samples, query_samples)| {
            train_data.extend(train_samples);
            query_data.extend(query_samples);
            (train_data, query_data)
        },
    );

    #[allow(clippy::tuple_array_conversions)]
    Ok([train_data, query_data])
}

/// Metadata for the `RadioML` data.
#[derive(Debug, Clone)]
pub struct RadioMLMetadata {
    /// Modulation mode.
    pub(crate) modulation: ModulationMode,
    /// Signal-to-noise ratio in dB.
    snr: Snr,
}

impl RadioMLMetadata {
    /// Returns the most common modulation mode in the given slice of metadata.
    #[allow(clippy::cast_possible_truncation, dead_code)]
    pub fn majority_modulation_mode(modes: &[&Self]) -> ModulationMode {
        let mut counts = [0; ModulationMode::variant_count()];

        for mode in modes {
            counts[mode.modulation.to_index() as usize] += 1;
        }

        let (majority, _) = abd_clam::utils::arg_max(&counts)
            .unwrap_or_else(|| unreachable!("We have 24 modulation modes."));

        ModulationMode::from_index(majority as u16).unwrap_or_else(|_| {
            unreachable!("We are only undoing what we did before. {}", majority)
        })
    }
}

impl Instance for RadioMLMetadata {
    fn to_bytes(&self) -> Vec<u8> {
        self.modulation
            .to_bytes()
            .into_iter()
            .chain(self.snr.to_bytes())
            .collect()
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, String>
    where
        Self: Sized,
    {
        let modulation = ModulationMode::from_bytes(&bytes[..2])?;
        let snr = Snr::from_bytes(&bytes[2..])?;
        Ok(Self { modulation, snr })
    }

    fn type_name() -> String {
        "RadioMLMetadata".to_string()
    }
}
