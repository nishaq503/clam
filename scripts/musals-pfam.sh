#!/bin/bash

# This script performs multiple sequence alignment using the CLAM Shell tool on the Pfam dataset.

# Usage: musals-pfam.sh <DATA_DIR> <DATA_SIZE> <SEED> <COST_MATRIX> <NUM_SAMPLES>
#
# Required arguments:
#   DATA_DIR        Path to directory containing input FASTA files
#   DATA_SIZE       Dataset size identifier (e.g., 10k, 100k)
#   SEED            Random seed for reproducibility
#   COST_MATRIX     Cost matrix type for alignment (e.g., affine, blosum62)
#   NUM_SAMPLES     Number of samples for evaluation
#
# Example usage:
#   ./scripts/musals-pfam.sh ../data/string-data/pfam-a 10k 42 affine 100

# Set up arguments from command line
DATA_DIR="${1:?DATA_DIR is required}"
DATA_SIZE="${2:?DATA_SIZE is required}"
SEED="${3:?SEED is required}"
COST_MATRIX="${4:?COST_MATRIX is required}"
NUM_SAMPLES="${5:?NUM_SAMPLES is required}"

# Derive other variables from the arguments
LOG_SUFFIX="pfam_${DATA_SIZE}"
INPUT_FASTA="${DATA_DIR}/pfam_${DATA_SIZE}.fasta"
BUILD_OUT_DIR="${DATA_DIR}/msa-results/${DATA_SIZE}"
TREE_NAME="tree-pfam_${DATA_SIZE}-string-levenshtein-span-reduction-factor::sqrt2.tree.bin"
ALIGNED_TREE_NAME="aligned-${COST_MATRIX}-${TREE_NAME}"

# Install the CLAM Shell binary
echo "Installing CLAM Shell..."
cargo install --path ./crates/abd-clam --features=shell --bin clam-shell
echo "CLAM Shell installed successfully."

# Build a tree for the pfam-10k dataset
echo "Building the tree..."
clam-shell \
    --inp-path "${INPUT_FASTA}" \
    --out-dir "${BUILD_OUT_DIR}" \
    --seed "${SEED}" \
    --log-name "build_${LOG_SUFFIX}" \
    build \
    --data-type string \
    --metric levenshtein
echo "Tree built successfully."

# Align the sequences using the built tree and write the results to a fasta file
echo "Aligning the sequences..."
clam-shell \
    --inp-path "${BUILD_OUT_DIR}/trees/${TREE_NAME}" \
    --out-dir "${BUILD_OUT_DIR}/aligned-trees" \
    --seed "${SEED}" \
    --log-name "musals_align_${LOG_SUFFIX}" \
    musals \
    align \
    --cost-matrix "${COST_MATRIX}" \
    --write-fasta
echo "Sequences aligned successfully."

# Evaluate the alignment
echo "Evaluating the alignment..."
clam-shell \
    --inp-path "${BUILD_OUT_DIR}/aligned-trees/${ALIGNED_TREE_NAME}" \
    --out-dir "${BUILD_OUT_DIR}/eval-results" \
    --seed "${SEED}" \
    --log-name "musals_eval_${LOG_SUFFIX}" \
    musals \
    evaluate \
    --num-samples "${NUM_SAMPLES}"
echo "Alignment evaluation completed successfully."
