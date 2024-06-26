//! Reproduce the CHAODA results in Rust.

use std::path::Path;

use abd_clam::{
    chaoda::{Chaoda, Vertex},
    Cluster, Dataset, PartitionCriteria, VecDataset,
};
use distances::Number;
use ndarray::prelude::*;
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

    if args.len() != 2 {
        return Err(format!(
            "Expected single input parameter, the directory where dataset will be read from. Got {args:?} instead",
        )
        );
    }

    // Parse args[0] into path object
    let data_dir = Path::new(&args[1]);
    let data_dir = std::fs::canonicalize(data_dir).map_err(|e| e.to_string())?;
    println!("Reading datasets from: {data_dir:?}");

    // Read the datasets and assign the metrics
    let seed = Some(42);
    let criteria = PartitionCriteria::default();
    let (train_datasets, roots): (Vec<_>, Vec<_>) = {
        let data = data::Data::read_paper_train(&data_dir)
            .into_iter()
            .map(|(name, (train, labels))| {
                let train = VecDataset::new(format!("{name}-manhattan"), train, manhattan, false);
                let train = train.assign_metadata(labels)?;
                let train_2 =
                    train.clone_with_new_metric(euclidean, false, format!("{name}-euclidean"));
                Ok(vec![train, train_2])
            })
            .collect::<Result<Vec<_>, String>>()?;

        data.into_iter()
            .flatten()
            .map(|mut d| {
                let root = Vertex::<f32>::new_root(&d, seed).partition(&mut d, &criteria, seed);
                (d, root)
            })
            .unzip()
    };
    println!("Training datasets:");
    for (d, r) in train_datasets.iter().zip(roots.iter()) {
        println!("{}: {}", d.name(), r.name());
    }

    // Train the CHAODA model
    let num_epochs = 10;
    let mut model = Chaoda::default();
    let mut training_data = None;
    for e in 0..num_epochs {
        println!("Starting Outer Epoch {}/{num_epochs}", e + 1);

        for (d, root) in train_datasets.iter().zip(roots.iter()) {
            let labels = d.metadata();
            training_data = Some(model.train(d, root, labels, 2, training_data));
        }
    }

    // Print the ROC scores for the training datasets

    // Run inference on the training datasets
    let predictions = train_datasets
        .iter()
        .zip(roots.iter())
        .map(|(d, root)| {
            let permutation = d
                .permuted_indices()
                .ok_or_else(|| "No permutation found".to_string())?;

            // Reorder the labels to match the original dataset
            let labels = d.metadata().to_vec();
            let labels = permutation
                .iter()
                .map(|&i| labels[i])
                .collect::<Vec<bool>>();

            // Reorder the predictions to match the original dataset
            let predictions = model
                .predict(d, root)
                .columns()
                .into_iter()
                .map(|v| v.to_owned().to_vec())
                .collect::<Vec<_>>();
            let n_rows = predictions.len();
            let predictions = permutation
                .iter()
                .flat_map(|&i| predictions[i].clone())
                .collect::<Vec<_>>();
            let predictions = Array2::from_shape_vec((n_rows, labels.len()), predictions)
                .map_err(|_| "Invalid shape".to_string())?;
            Ok((d.name(), labels, predictions))
        })
        .collect::<Result<Vec<_>, String>>()?;

    // Compute the ROC scores
    let roc_scores = predictions
        .chunks(2)
        .map(|chunk| {
            let chunk: [_; 2] = chunk
                .to_vec()
                .try_into()
                .map_err(|_| "Invalid chunk size".to_string())?;
            let [(name, labels, l1_pred), (_, _, l2_pred)] = chunk;
            let name = name
                .split('-')
                .next()
                .ok_or_else(|| "Invalid dataset name".to_string())?;
            let predictions = ndarray::concatenate![Axis(0), l1_pred, l2_pred];
            let predictions = Chaoda::aggregate_predictions(&predictions);
            let y_true = labels
                .iter()
                .map(|&l| if l { 1.0 } else { 0.0 })
                .collect::<Vec<f32>>();
            let roc_score = roc_auc_score(&y_true, &predictions).as_f32();
            Ok((name.to_string(), roc_score))
        })
        .collect::<Result<Vec<_>, String>>()?;
    println!("ROC scores for Training datasets:");
    for (name, score) in roc_scores {
        println!("{name}: {score:.6}");
    }

    // Print the ROC scores for the inference datasets

    // Read the inference datasets
    let (infer_datasets, roots): (Vec<_>, Vec<_>) = {
        let data = data::Data::read_paper_inference(&data_dir)
            .into_iter()
            .map(|(name, (infer, labels))| {
                let infer = VecDataset::new(format!("{name}-manhattan"), infer, manhattan, false);
                let infer = infer.assign_metadata(labels)?;
                let infer_2 =
                    infer.clone_with_new_metric(euclidean, false, format!("{name}-euclidean"));
                Ok(vec![infer, infer_2])
            })
            .collect::<Result<Vec<_>, String>>()?;

        data.into_iter()
            .flatten()
            .map(|mut d| {
                let root = Vertex::<f32>::new_root(&d, seed).partition(&mut d, &criteria, seed);
                (d, root)
            })
            .unzip()
    };

    // Run inference on the inference datasets
    let predictions = infer_datasets
        .iter()
        .zip(roots.iter())
        .map(|(d, root)| {
            let permutation = d
                .permuted_indices()
                .ok_or_else(|| "No permutation found".to_string())?;

            // Reorder the labels to match the original dataset
            let labels = d.metadata().to_vec();
            let labels = permutation
                .iter()
                .map(|&i| labels[i])
                .collect::<Vec<bool>>();

            // Reorder the predictions to match the original dataset
            let predictions = model
                .predict(d, root)
                .columns()
                .into_iter()
                .map(|v| v.to_owned().to_vec())
                .collect::<Vec<_>>();
            let n_rows = predictions.len();
            let predictions = permutation
                .iter()
                .flat_map(|&i| predictions[i].clone())
                .collect::<Vec<_>>();
            let predictions = Array2::from_shape_vec((n_rows, labels.len()), predictions)
                .map_err(|_| "Invalid shape".to_string())?;

            Ok((d.name(), labels, predictions))
        })
        .collect::<Result<Vec<_>, String>>()?;

    // Compute the ROC scores
    let roc_scores = predictions
        .chunks(2)
        .map(|chunk| {
            let chunk: [_; 2] = chunk
                .to_vec()
                .try_into()
                .map_err(|_| "Invalid chunk size".to_string())?;
            let [(name, labels, l1_pred), (_, _, l2_pred)] = chunk;
            let name = name
                .split('-')
                .next()
                .ok_or_else(|| "Invalid dataset name".to_string())?;
            let predictions = ndarray::concatenate![Axis(0), l1_pred, l2_pred];
            let predictions = Chaoda::aggregate_predictions(&predictions);
            let y_true = labels
                .iter()
                .map(|&l| if l { 1.0 } else { 0.0 })
                .collect::<Vec<f32>>();
            let roc_score = roc_auc_score(&y_true, &predictions).as_f32();
            Ok((name.to_string(), roc_score))
        })
        .collect::<Result<Vec<_>, String>>()?;
    println!("ROC scores for Inference datasets:");
    for (name, score) in roc_scores {
        println!("{name}: {score:.6}");
    }

    Ok(())
}
