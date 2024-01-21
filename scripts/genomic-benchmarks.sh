#!/bin/bash

# Requires:
# - Rust version 1.72.0 or higher: https://www.rust-lang.org/tools/install
# - Ann Datasets: https://github.com/erikbern/ann-benchmarks#data-sets

# Set input and output directories
input_dir="../data/silva"
output_dir="../data/silva/knn-reports"

echo "Starting genomic-benchmarks at: $(date)"

# Compile cakes-results
cargo build --release

for sample_size in 1000 2000 5000 10000 50000 100000 250000
do
    ./target/release/cakes-results \
        --input-dir $input_dir \
        --output-dir $output_dir \
        genomic \
        --sample-size $sample_size \
        --ks 10 100 \
        --rs 25 100 250
done

./target/release/cakes-results \
    --input-dir $input_dir \
    --output-dir $output_dir \
    genomic \
    --ks 10 100 \
    --rs 25 100 250
