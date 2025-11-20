# CLAM Shell

## Quickstart

```bash
# 1. Create an ignored directory to experiment with.
mkdir -p target/experiments

# 2. Generate 50 small random vectors of dimension 10, partitioned into 90% train and 10% test sets.
cargo run --package shell -- \
    --out-path target/experiments/data/small-vectors.npy \
    --seed 42 \
    --log-name gen-data.log \
    generate-data generate \
        --num-vectors 50 \
        --dimensions 10 \
        --data-type f32 \
        --partitions 90,10 \
        --min-val 0.0 \
        --max-val 1.0

# 3. Build the tree
cargo run --package shell -- \
    --inp-path target/experiments/data/small-vectors-45.npy \
    --out-path target/experiments/small-trees \
    --metric euclidean \
    --log-name cakes-build.log \
    cakes build

# 4. Search for some queries.
cargo run --package shell -- \
    --inp-path target/experiments/small-trees \
    --out-path target/experiments/small-results.json \
    --log-name cakes-search.log \
    cakes search \
        --queries-path ./target/experiments/data/small-vectors-5.npy \
        --cakes-algorithms knn-linear:k=2 \
        --cakes-algorithms knn-linear:k=5
```
