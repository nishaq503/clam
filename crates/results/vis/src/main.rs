//! Let's look at some visualizations!

use std::{fs::OpenOptions, path::Path};

use abd_clam::{
    chaoda::{Chaoda, Vertex},
    dim_red::MassSpringSystem,
    Cluster, PartitionCriteria, VecDataset,
};
use csv::WriterBuilder;
use mt_logger::{mt_flush, mt_log, mt_new, Level, OutputStream};
use results_chaoda::Data;

pub fn write_results(outpath: &Path, results: &Vec<String>) {
    let file = OpenOptions::new().create(true).append(true).open(outpath);
    if let Ok(file) = file {
        // Create a CSV writer
        let mut writer = WriterBuilder::new()
            .delimiter(b',') // Set delimiter (optional)
            .from_writer(file);

        // Write the data to the CSV file
        let _ = writer.write_record(results);

        // Flush and close the writer to ensure all data is written
        let _ = writer.flush();
    }
}

fn main() -> Result<(), String> {
    mt_new!(None, Level::Debug, OutputStream::StdOut);

    let args: Vec<String> = std::env::args().collect();

    if args.len() != 3 {
        let msg =
            format!("Expected one input parameter, the input directory and the output directory. Got {args:?} instead",);
        return Err(msg);
    }

    // Parse args[1] into path object
    let inp_dir = Path::new(&args[1]);
    let inp_dir = std::fs::canonicalize(inp_dir).map_err(|e| e.to_string())?;
    mt_log!(Level::Info, "Reading datasets from: {inp_dir:?}");

    // Parse args[2] into path object
    let out_dir = Path::new(&args[2]);
    let out_dir = std::fs::canonicalize(out_dir).map_err(|e| e.to_string())?;
    mt_log!(Level::Info, "Writing results to: {out_dir:?}");

    // Load the model
    let model_path = inp_dir.join("pre-trained-chaoda-model.bin");
    if !model_path.exists() {
        return Err(format!(
            "Pre-trained model not found at {model_path:?}. Please train the model first."
        ));
    }
    let model = Chaoda::load(&model_path)?;

    // Load the data
    let datasets: &[&str] = &[
        "cardio",
        "arrhythmia",
        "satellite",
        "mnist", // Sometimes produces infinte forces from springs.
    ];

    let metric = |x: &Vec<f32>, y: &Vec<f32>| distances::vectors::euclidean::<_, f32>(x, y);
    let seed = Some(42);
    let criteria = PartitionCriteria::default();

    for &name in datasets {
        mt_log!(Level::Info, "Processing dataset: {name}");

        let (data, root) = {
            let (data, labels) = Data::new(name)?.read(&inp_dir)?;
            let mut data = VecDataset::new(name.to_string(), data, metric, false).assign_metadata(labels)?;

            mt_log!(Level::Info, "Building the tree for {name}");
            let root = Vertex::new_root(&data, seed).partition(&mut data, &criteria, seed);

            (data, root)
        };

        mt_log!(Level::Info, "Creating graphs for {name}");
        let named_graphs = model
            .create_graphs(&data, &root)
            .into_iter()
            .zip(model.predictor_names())
            .flat_map(|(m_graphs, (member, ml_models))| {
                m_graphs
                    .into_iter()
                    .zip(ml_models)
                    .map(move |(graph, ml_model)| (ml_model, member.clone(), graph))
            })
            .collect::<Vec<_>>();

        let save_intermediates = true;

        for (ml_model, member, graph) in named_graphs {
            let reduced_name = format!("{name}-{member}-{ml_model}");

            mt_log!(Level::Info, "Simulating {reduced_name}");
            let (graph_dir, steps_dir) = {
                let graph_dir = out_dir.join(&reduced_name);
                // If the directory exists, delete it
                if graph_dir.exists() {
                    std::fs::remove_dir_all(&graph_dir).map_err(|e| e.to_string())?;
                }
                std::fs::create_dir(&graph_dir).map_err(|e| e.to_string())?;

                let steps_dir = graph_dir.join("steps");
                std::fs::create_dir(&steps_dir).map_err(|e| e.to_string())?;

                (graph_dir, steps_dir)
            };

            let mss = {
                let mss = MassSpringSystem::<_, 3>::from_graph(&graph, 1.0, 0.99, seed);

                if save_intermediates {
                    let descriptors = vec![
                        "root_cardinality".to_string(),
                        "tree_height".to_string(),
                        "graph_mass_cardinality".to_string(),
                        "graph_edge_cardinality".to_string(),
                    ];

                    let descriptor_data = vec![
                        root.cardinality().to_string(),
                        root.max_leaf_depth().to_string(),
                        mss.masses().len().to_string(),
                        mss.springs().len().to_string(),
                    ];

                    let descriptor_path = &graph_dir.join("descriptor.txt");

                    write_results(descriptor_path, &descriptors);
                    write_results(descriptor_path, &descriptor_data);
                }

                if save_intermediates {
                    mss.evolve_with_saves(0.01, 1_000, 1, &data, &steps_dir, &reduced_name)?
                } else {
                    mss.evolve_to_stability(0.01, 50)
                }
            };

            mt_log!(Level::Info, "Writing logs for {reduced_name}");
            let logs = mss.logs().iter().map(|r| r.to_vec()).collect::<Vec<_>>();
            let logs = VecDataset::new(reduced_name.clone(), logs, metric, false);
            logs.to_npy(&graph_dir.join("logs.npy"))?;

            mt_log!(Level::Info, "Writing {reduced_name}");
            let reduced_data = mss.get_reduced_embedding(&data, &reduced_name);
            reduced_data.to_npy(&graph_dir.join("final.npy"))?;
        }
    }

    mt_flush!().map_err(|e| e.to_string())?;

    Ok(())
}
