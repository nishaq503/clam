#!/bin/bash

# Set root directory for the data
data_root="../data/ann-benchmarks"

# Set input and output directories in terms of the root directory
data_dir="$data_root/datasets"
cakes_dir="$data_root/scaling-reports"
output_dir="$data_root/rival-reports"

echo "Starting rivals' benchmarks at: $(date)"

if test -d ./.venv; then
    echo "Using existing virtual environment"
else
    echo "Creating new virtual environment"
    ~/.pyenv/versions/3.9.16/bin/python -m venv .venv
fi

source .venv/bin/activate

# Install cakes-results
cd cakes-results
pip install -e .
cd ..

# Run rivals
for dataset in "glove-25"
do
    for metric in "cosine"
    do
        for rival in "faiss-ivf-flat" "annoy"
        do
            python -m cakes_results rivals run-rival \
                -r $rival \
                -d $data_dir \
                -n $dataset \
                -m $metric \
                -c $cakes_dir \
                -o $output_dir
        done
    done
done

echo "Finished rivals' benchmarks at: $(date)"

deactivate
