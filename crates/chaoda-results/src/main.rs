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
    let (train_datasets, roots): (Vec<_>, Vec<_>) = {
        let data = data::Data::read_paper_train(&data_dir)?
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

    let model = if use_pre_trained {
        // Load the pre-trained CHAODA model
        println!("Loading pre-trained model from: {model_path:?}");
        Chaoda::load(&model_path)?
    } else {
        // Train the CHAODA model
        let num_epochs_outer = 10;
        let num_epochs_inner = 2;
        let mut model = Chaoda::default();
        let mut training_data = None;
        for e in 0..num_epochs_outer {
            println!("Starting Outer Epoch {}/{num_epochs_outer}", e + 1);

            for (d, root) in train_datasets.iter().zip(roots.iter()) {
                println!("Training on dataset: {}", d.name());

                let labels = d.metadata();
                training_data = Some(model.train(d, root, labels, num_epochs_inner, training_data));
            }
        }
        println!("Training complete. Saving model to: {model_path:?}");
        model.save(&model_path)?;
        model
    };

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
            let mut labels = permutation
                .iter()
                .zip(labels)
                .map(|(&i, l)| (i, l))
                .collect::<Vec<_>>();
            labels.sort_unstable_by_key(|(i, _)| *i);
            let labels = labels.into_iter().map(|(_, l)| l).collect::<Vec<_>>();

            // Reorder the predictions to match the original dataset
            let predictions = model.predict(d, root);
            assert_eq!(
                predictions.ncols(),
                labels.len(),
                "Number of predictions do not match the number of labels."
            );
            let mut predictions = predictions
                .columns()
                .into_iter()
                .map(|v| v.to_owned().to_vec())
                .zip(permutation.iter())
                .collect::<Vec<_>>();
            predictions.sort_unstable_by_key(|(_, &i)| i);
            let predictions = predictions
                .into_iter()
                .flat_map(|(v, _)| v)
                .collect::<Vec<_>>();
            assert_eq!(
                predictions.len(),
                model.num_predictors() * labels.len(),
                "Predictions do not match after inverting permutation."
            );
            let predictions =
                Array2::from_shape_vec((model.num_predictors(), labels.len()), predictions)
                    .map_err(|e| format!("Invalid shape: {e:?}"))?;
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
            let predictions = Chaoda::aggregate_predictions(&predictions).to_vec();
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
    let mut infer_labels = Vec::new();

    // Read the inference datasets
    let (infer_datasets, roots): (Vec<_>, Vec<_>) = {
        let data = data::Data::read_paper_inference(&data_dir)?
            .into_iter()
            .map(|(name, (infer, labels))| {
                // Note that we do not assign metadata here, so CHAODA will not have access to the labels
                // We push labels twice to match the number of distance metrics
                infer_labels.push(labels.clone());
                infer_labels.push(labels);
                let infer = VecDataset::new(format!("{name}-manhattan"), infer, manhattan, false);
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
        .zip(infer_labels)
        .map(|((d, root), labels)| {
            let permutation = d
                .permuted_indices()
                .ok_or_else(|| "No permutation found".to_string())?;

            // Reorder the predictions to match the original dataset
            let predictions = model.predict(d, root);
            assert_eq!(
                predictions.ncols(),
                labels.len(),
                "Number of predictions do not match the number of labels."
            );
            let mut predictions = predictions
                .columns()
                .into_iter()
                .map(|v| v.to_owned().to_vec())
                .zip(permutation.iter())
                .collect::<Vec<_>>();
            predictions.sort_unstable_by_key(|(_, &i)| i);
            let predictions = predictions
                .into_iter()
                .flat_map(|(v, _)| v)
                .collect::<Vec<_>>();
            assert_eq!(
                predictions.len(),
                model.num_predictors() * labels.len(),
                "Predictions do not match after inverting permutation."
            );
            let predictions =
                Array2::from_shape_vec((model.num_predictors(), labels.len()), predictions)
                    .map_err(|e| format!("Invalid shape: {e:?}"))?;
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
            let predictions = Chaoda::aggregate_predictions(&predictions).to_vec();
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
