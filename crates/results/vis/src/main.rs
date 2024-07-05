//! Let's look at some visualizations!

use std::path::Path;

use abd_clam::{
    chaoda::{Chaoda, Vertex},
    dim_red::MassSpringSystem,
    Cluster, PartitionCriteria, VecDataset,
};
use results_chaoda::Data;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() != 3 {
        let msg =
            format!("Expected one input parameter, the input directory and the output directory. Got {args:?} instead",);
        return Err(msg);
    }

    // Parse args[1] into path object
    let inp_dir = Path::new(&args[1]);
    let inp_dir = std::fs::canonicalize(inp_dir).map_err(|e| e.to_string())?;
    println!("Reading datasets from: {inp_dir:?}");

    // Parse args[2] into path object
    let out_dir = Path::new(&args[2]);
    let out_dir = std::fs::canonicalize(out_dir).map_err(|e| e.to_string())?;
    println!("Writing results to: {out_dir:?}");

    // Load the model
    let model_path = inp_dir.join("pre-trained-chaoda-model.bin");
    if !model_path.exists() {
        return Err(format!(
            "Pre-trained model not found at {model_path:?}. Please train the model first."
        ));
    }
    let model = Chaoda::load(&model_path)?;

    // Load the data
    let datasets: &[&str] = &["cardio"];

    let metric = |x: &Vec<f32>, y: &Vec<f32>| distances::vectors::euclidean::<_, f32>(x, y);
    let seed = Some(42);
    let criteria = PartitionCriteria::default();

    for name in datasets {
        println!("Processing dataset: {name}");

        let (data, root) = {
            let (data, labels) = Data::new(name)?.read(&inp_dir)?;
            let mut data = VecDataset::new(name.to_string(), data, metric, false).assign_metadata(labels)?;

            println!("Building the tree for {name}");
            let root = Vertex::new_root(&data, seed).partition(&mut data, &criteria, seed);

            (data, root)
        };

        println!("Creating graphs for {name}");
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

        for (ml_model, member, graph) in named_graphs {
            let reduced_name = format!("{name}_{member}_{ml_model}");

            println!("Simulating {reduced_name}");
            let mss = MassSpringSystem::<_, 3>::from_graph(&graph, 1.0, 0.99, seed);
            let mss = mss.evolve(0.1, 10_000);

            println!("Writing logs for {reduced_name}");
            let logs = mss.logs().iter().map(|r| r.to_vec()).collect::<Vec<_>>();
            let logs = VecDataset::new(reduced_name.clone(), logs, metric, false);
            logs.to_npy(&out_dir.join(format!("{reduced_name}_logs.npy")))?;

            println!("Writing {reduced_name}");
            let reduced_data = mss.get_reduced_embedding(&data, &reduced_name);
            reduced_data.to_npy(&out_dir.join(format!("{reduced_name}.npy")))?;
        }
    }

    Ok(())
}
