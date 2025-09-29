# Reproducible Benchmarks and Plots for CLAM research

## Requirements

The following commands should succeed from the workspace root directory:

```sh
cargo build --release --workspace --all-features
uv sync --all-packages
```

The tests should also be successful:

```sh
cargo test --release --workspace --all-features
uv run pytest
```

# Usage

TODO
