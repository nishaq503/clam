//! Reproduce the CHAODA results in Rust.

use std::path::Path;

use abd_clam::{
    chaoda::{Chaoda, Vertex},
    Cluster, Dataset, PartitionCriteria, VecDataset,
};
use smartcore::metrics::roc_auc_score;

mod data;

#[allow(clippy::ptr_arg)]
fn euclidean(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    distances::vectors::euclidean(x, y)
}

#[allow(clippy::ptr_arg)]
fn manhattan(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    distances::vectors::manhattan(x, y)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() != 3 {
        return Err(format!(
            "Expected two input parameters, the directory where dataset will be read from and whether to use a pre-trained model (if found). Got {args:?} instead",
        )
        );
    }

    // Parse args[1] into path object
    let data_dir = Path::new(&args[1]);
    let data_dir = std::fs::canonicalize(data_dir).map_err(|e| e.to_string())?;
    println!("Reading datasets from: {data_dir:?}");

    // Parse args[2] into boolean
    let use_pre_trained = args[2].parse::<String>().map_err(|e| e.to_string())?;
    let use_pre_trained = {
        match use_pre_trained.as_str() {
            "true" => true,
            "false" => false,
            _ => {
                return Err(format!(
                "Invalid value for use_pre_trained: {use_pre_trained}. Expected 'true' or 'false'"
            ))
            }
        }
    };
    // Build path to pre-trained model
    let model_path = data_dir.join("pre-trained-chaoda-model.bin");
    let use_pre_trained = use_pre_trained && model_path.exists();

    // Set some parameters for tree building
    let seed = Some(42);
    let criteria = PartitionCriteria::default();

    // Read the datasets and assign the metrics
    let train_datasets = {
        let datasets = data::Data::read_paper_train(&data_dir)?
            .into_iter()
            .map(|(name, (train, labels))| {
                let train = VecDataset::new(format!("{name}-manhattan"), train, manhattan, false);
                let train = train.assign_metadata(labels)?;
                let train_2 =
                    train.clone_with_new_metric(euclidean, false, format!("{name}-euclidean"));
                Ok(vec![train, train_2])
            })
            .collect::<Result<Vec<_>, String>>()?;
        let datasets = datasets.into_iter().flatten().collect::<Vec<_>>();
        datasets
            .into_iter()
            .map(|d| {
                let labels = d.metadata().to_vec();
                (d, labels)
            })
            .collect::<Vec<_>>()
    };
    println!("Training datasets:");
    for (d, _) in train_datasets.iter() {
        println!("{}", d.name());
    }

    let model = if use_pre_trained {
        // Load the pre-trained CHAODA model
        println!("Loading pre-trained model from: {model_path:?}");
        Chaoda::load(&model_path)?
    } else {
        // Train the CHAODA model
        let num_epochs = 10;
        let mut model = Chaoda::default();
        model.train::<_, _, _, Vertex<_>, 3, _>(train_datasets, num_epochs, &criteria, None, seed);
        println!("Training complete");
        model.save(&model_path)?;
        println!("Model saved to: {model_path:?}");
        model
    };

    // Print the ROC scores for all datasets
    for (name, (data, labels)) in data::Data::read_all(&data_dir)? {
        println!("Starting evaluation for: {name}");

        let min_cardinality = if data.len() < 10_000 {
            1
        } else if data.len() < 40_000 {
            4
        } else if data.len() < 100_000 {
            8
        } else {
            16
        };
        println!("Using min_cardinality: {min_cardinality}");
        let criteria = PartitionCriteria::default().with_min_cardinality(min_cardinality);

        let mut l1_data = VecDataset::new(name.clone(), data, manhattan, false);
        let l1_root = Vertex::new_root(&l1_data, seed).partition(&mut l1_data, &criteria, seed);
        let l1_scores = Chaoda::aggregate_predictions(&model.predict(&l1_data, &l1_root));

        let mut l2_data =
            l1_data.clone_with_new_metric(euclidean, false, format!("{name}-euclidean"));
        let l2_root = Vertex::new_root(&l2_data, seed).partition(&mut l2_data, &criteria, seed);
        let l2_scores = Chaoda::aggregate_predictions(&model.predict(&l2_data, &l2_root));

        let y_pred = l1_scores
            .into_iter()
            .zip(l2_scores)
            .map(|(l1, l2)| (l1 + l2) / 2.0)
            .collect::<Vec<_>>();
        let y_true = labels
            .iter()
            .map(|&l| if l { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let roc_auc = roc_auc_score(&y_true, &y_pred);
        println!("{name}: Aggregate {roc_auc:.6}");
    }

    Ok(())
}
