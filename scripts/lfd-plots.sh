#!/bin/bash

# Requires:
# - Rust version 1.72.0 or higher: https://www.rust-lang.org/tools/install
# - Ann Datasets: https://github.com/erikbern/ann-benchmarks#data-sets

# Set input and output directories
data_dir="../data"

# Set output directory from the data_dir
output_dir="$data_dir/lfd-exploration/data"

echo "Starting lfd-plots at: $(date)"

# Compile cakes-results
cargo build --release

# # Create Silva csv files
# # for cardinality in 1000 2000 5000 10000 50000 100000 250000 2224640
# for cardinality in 2224640
# do
#     ./target/release/cakes-results \
#         --input-dir "$data_dir/silva/silva-SSU-Ref-$cardinality-cakes" \
#         --output-dir $output_dir \
#         lfd-plots \
#         --dataset "silva"
# done

# # Create RadioML csv files
# # for cardinality in 768 3972 12288 49152 97920
# for cardinality in 97920
# do
#     ./target/release/cakes-results \
#         --input-dir "$data_dir/radio-ml/output/cakes-snr-30-cardinality-$cardinality" \
#         --output-dir $output_dir \
#         lfd-plots \
#         --dataset "radioml"
# done

# # Create AnnDatasets csv files
# for dataset in "fashion-mnist" "glove-25" "sift" "random-1000000-128"
# do
#     ./target/release/cakes-results \
#         --input-dir "$data_dir/ann-benchmarks/datasets" \
#         --output-dir $output_dir \
#         lfd-plots \
#         --dataset $dataset
# done

# Create the virtual environment if it doesn't exist
if test -d ./.venv; then
    echo "Using existing virtual environment"

    # Activate the virtual environment
    source .venv/bin/activate
else
    echo "Creating new virtual environment"
    ~/.pyenv/versions/3.9.16/bin/python -m venv .venv

    # Activate the virtual environment
    source .venv/bin/activate

    # Install cakes-results
    cd cakes-results
    pip install -e .
    cd ..
fi

# Create plots
python -m cakes_results lfd-plots create-plots \
    -i ../data/lfd-exploration/data \
    -o ../data/lfd-exploration/plots

echo "Finished lfd-plots at: $(date)"

# Deactivate the virtual environment
deactivate
