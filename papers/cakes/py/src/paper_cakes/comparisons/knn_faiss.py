"""Running FAISS KNN algorithms for comparison against CAKES."""

import os
import pathlib
import time

import faiss
import numpy
import typer

from paper_cakes.datasets import Dataset
from paper_cakes.datasets import Metric
from paper_cakes.utils import configure_logging


def faiss_flat(  # noqa: PLR0913, C901, PLR0915
    inp_dir: pathlib.Path = typer.Option(  # noqa: B008
        ...,
        "-i",
        "--inp-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="The input directory containing ann-benchmarks datasets.",
    ),
    out_dir: pathlib.Path | None = typer.Option(  # noqa: B008
        None,
        "-o",
        "--out-dir",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        help="Directory to save results",
    ),
    log_dir: pathlib.Path | None = typer.Option(  # noqa: B008
        None,
        "-l",
        "--log-dir",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        help="Directory to save logs",
    ),
    data_name: str = typer.Option(
        ...,
        "-d",
        "--data-name",
        help="Name of the dataset to process.",
    ),
    seed: int | None = typer.Option(
        None,
        "-s",
        "--seed",
        help="Random seed for reproducibility",
    ),
    k: int = typer.Option(
        10,
        "-k",
        "--num-neighbors",
        help="Number of nearest neighbors to retrieve",
    ),
    measurement_time: float = typer.Option(
        5.0,
        "-t",
        "--measurement-time",
        help="The minimum time (in seconds) to run the search algorithm.",
    ),
) -> None:
    """Uses a FAISS FLAT index to compute KNN results for comparison against CAKES."""
    typer.echo("Running FAISS-FLAT benchmarks...")

    if out_dir is None:
        # Check if `inp_dir` is writable
        if not os.access(inp_dir, os.W_OK):
            raise PermissionError(f"Cannot write to directory: {inp_dir}")
        out_dir = inp_dir / "faiss_flat_results"
    out_dir.mkdir(parents=False, exist_ok=True)

    if log_dir is None:
        # Check if `inp_dir` is writable
        if not os.access(inp_dir, os.W_OK):
            raise PermissionError(f"Cannot write to directory: {inp_dir}")
        log_dir = inp_dir / "faiss_flat_logs"
    log_dir.mkdir(parents=False, exist_ok=True)

    # Create a logger
    logger = configure_logging("faiss_flat", file_path=log_dir / "faiss_flat.log")
    logger.info("-" * 120)  # Separator line in log file

    # Create random number generator
    rng = None
    if seed is not None:
        rng = numpy.random.default_rng(seed)
        logger.info(f"Using random seed: {seed}")

    # Load dataset
    dataset = Dataset.from_name(data_name)
    logger.info(f"Processing dataset: {dataset.value}")
    train_data = dataset.read_train(inp_dir, rng)
    test_data = dataset.read_test(inp_dir, rng)
    metric = dataset.metric()
    logger.info(f"Dataset metric: {metric.name}")

    # Create FAISS index
    logger.info("Creating FAISS-FLAT index...")
    dim = train_data.shape[1]
    if metric == Metric.Euclidean:
        index = faiss.IndexFlatL2(dim)
    elif metric == Metric.Cosine:
        logger.info("Normalizing data for cosine similarity...")
        faiss.normalize_L2(train_data)
        faiss.normalize_L2(test_data)
        index = faiss.IndexFlatIP(dim)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    logger.info(f"Created FAISS index. is_trained={index.is_trained}")

    logger.info("Adding training data to FAISS index...")
    index.add(train_data)
    logger.info(f"Number of vectors in index: {index.ntotal}")

    # Perform search
    logger.info("Starting KNN search...")
    distances: numpy.ndarray
    indices: numpy.ndarray

    start_time = time.perf_counter()
    distances, indices = index.search(test_data, k)
    elapsed_time = time.perf_counter() - start_time
    num_runs = 1
    if elapsed_time < measurement_time:
        logger.info("Continuing KNN search to meet measurement time...")
        while elapsed_time < measurement_time:
            distances, indices = index.search(test_data, k)
            num_runs += 1
            elapsed_time = time.perf_counter() - start_time

    if indices.shape != (test_data.shape[0], k):
        raise ValueError(f"Unexpected indices shape: {indices.shape}")
    if distances.shape != (test_data.shape[0], k):
        raise ValueError(f"Unexpected distances shape: {distances.shape}")

    total_time = time.perf_counter() - start_time
    time_per_run = total_time / num_runs
    throughput = test_data.shape[0] / time_per_run
    logger.info(f"Completed KNN search in {total_time:.6f} seconds over {num_runs} runs.")
    logger.info(f"Throughput: {throughput:.2e} queries/second.")

    # Save results
    out_path_indices = out_dir / f"{dataset.value}_faiss_flat_indices.npy"
    out_path_distances = out_dir / f"{dataset.value}_faiss_flat_distances.npy"
    numpy.save(out_path_indices, indices)
    numpy.save(out_path_distances, distances)
    logger.info(f"Saved indices to: {out_path_indices}")
    logger.info(f"Saved distances to: {out_path_distances}")
    logger.info("FAISS-FLAT benchmarks complete.")


__all__ = ["faiss_flat"]
