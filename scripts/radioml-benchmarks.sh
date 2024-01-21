#!/bin/bash

# Requires:
# - Rust version 1.72.0 or higher: https://www.rust-lang.org/tools/install
# - Ann Datasets: https://github.com/erikbern/ann-benchmarks#data-sets

# Set data directory: "/path/to/data/radio-ml"
data_dir="../data/radio-ml"

# Set input and output directories from the data directory
input_dir="$data_dir/input"
output_dir="$data_dir/output"

echo "Starting radioml-benchmarks at: $(date)"

# Compile cakes-results
cargo build --release

# for sample_size in 128 256 512 1024
for sample_size in 2048 4080
do
    # for snr in -10 0 10 20 30
    for snr in 10
    do
        ./target/release/cakes-results \
            --input-dir $input_dir \
            --output-dir $output_dir \
            radio \
            --snr $snr \
            --sample-size $sample_size \
            --num-queries 16 \
            --ks 10 100
    done
done

# cakes with sample size 2048 took 4523.0 seconds
