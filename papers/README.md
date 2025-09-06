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

## Running the Notebooks

In VSCode, install the "Jupyter" extension, then open a notebook in the `notebooks` directory. Running the `build` and `sync` commands above should have created the Python virtual environment. Select the Python interpreter from the virtual environment as the kernel for the notebook (there should be a button in the top right of the notebook interface to select the kernel). You should then be able to run the notebook cells and interact with the plots.

# Usage

TODO
