#!/bin/bash

# Requires:
# - Rust version 1.72.0 or higher: https://www.rust-lang.org/tools/install
# - Ann Datasets: https://github.com/erikbern/ann-benchmarks#data-sets

# Set input and output directories
input_dir="../data/ann-benchmarks/datasets"
output_dir="../data/ann-benchmarks/scaling-reports"

echo "Starting scaling-benchmarks at: $(date)"

# Compile cakes-results
cargo build --release

# for dataset in "sift" "gist" "deep-image" "random-1000000-128-euclidean"
# for dataset in "fashion-mnist" "mnist" "glove-25" "glove-100"
for dataset in "random-1000000-128-euclidean"
do
    ./target/release/cakes-results \
        --input-dir $input_dir \
        --output-dir $output_dir \
        scaling \
        --dataset $dataset \
        --max-memory 100
done
