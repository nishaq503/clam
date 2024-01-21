"""Utility functions for the algorithms."""


import json
import pathlib

from . import base


def save_knn_results(
    output_dir: pathlib.Path,
    dataset_name: str,
    metric: base.Metric,
    scale: int,
    k: int,
    knn_results: list[list[tuple[int, float]]],
) -> None:
    """Save the ground truth to the output directory.

    Args:
        output_dir: Output directory.
        dataset_name: Name of the dataset.
        metric: Distance metric used for the search.
        scale: Scale of the dataset.
        k: Number of nearest neighbors to find.
        knn_results: List of lists of tuples of (index, distance) of the
        nearest neighbors of each query.
    """
    knn_path = (
        output_dir / f"ground-truth-knn-{k}-{dataset_name}-{scale}-{metric.value}.json"
    )
    with knn_path.open("w") as file:
        json.dump(knn_results, file, indent=2)


def save_rnn_results(
    output_dir: pathlib.Path,
    dataset_name: str,
    metric: base.Metric,
    scale: int,
    radius: float,
    rnn_results: list[list[tuple[int, float]]],
) -> None:
    """Save the ground truth to the output directory.

    Args:
        output_dir: Output directory.
        dataset_name: Name of the dataset.
        metric: Distance metric used for the search.
        scale: Scale of the dataset.
        radius: Radius of the search.
        rnn_results: List of lists of tuples of (index, distance) of the
        neighbors of each query.
    """

    # Save the rnn ground truth
    rnn_path = (
        output_dir
        / f"ground-truth-rnn-{radius}-{dataset_name}-{scale}-{metric.value}.json"
    )
    with rnn_path.open("w") as file:
        json.dump(rnn_results, file, indent=2)


def load_knn_results(
    output_dir: pathlib.Path,
    dataset_name: str,
    metric: base.Metric,
    scale: int,
    k: int,
) -> list[list[tuple[int, float]]]:
    """Load the knn results from the output directory.

    Args:
        output_dir: Output directory.
        dataset_name: Name of the dataset.
        metric: Distance metric used for the search.
        scale: Scale of the dataset.
        k: Number of nearest neighbors to find.

    Returns:
        knn_results: Nearest neighbors for the knn search.
    """

    # Load the knn results
    knn_path = (
        output_dir / f"ground-truth-knn-{k}-{dataset_name}-{scale}-{metric.value}.json"
    )
    with knn_path.open("r") as file:
        knn_results = json.load(file)

    return knn_results


def load_rnn_results(
    output_dir: pathlib.Path,
    dataset_name: str,
    metric: base.Metric,
    scale: int,
    radius: float,
) -> list[list[tuple[int, float]]]:
    """Load the rnn results from the output directory.

    Args:
        output_dir: Output directory.
        dataset_name: Name of the dataset.
        metric: Distance metric used for the search.
        scale: Scale of the dataset.
        radius: Radius of the search.

    Returns:
        rnn_results: Neighbors for the rnn search.
    """

    # Load the rnn results
    rnn_path = (
        output_dir
        / f"ground-truth-rnn-{radius}-{dataset_name}-{scale}-{metric.value}.json"
    )
    with rnn_path.open("r") as file:
        rnn_results = json.load(file)

    return rnn_results


def compute_recall(
    true_results: list[list[tuple[int, float]]],
    results: list[list[tuple[int, float]]],
) -> float:
    """Compute the recall of the results.

    Args:
        true_results: The true results. A list of lists of (index, distance) tuples.
        results: The results. A list of lists of (index, distance) tuples.

    Returns:
        The recall of the results.
    """
    each_recall = [_recall(t, r) for t, r in zip(true_results, results)]
    return sum(each_recall) / len(each_recall)


def _recall(
    true_results: list[tuple[int, float]],
    results: list[tuple[int, float]],
) -> float:
    """Compute the recall of the results.

    Args:
        true_results: The true results. A list of (index, distance) tuples.
        results: The results. A list of (index, distance) tuples.

    Returns:
        The recall of the results.
    """

    if len(true_results) == 0:
        return 1.0 if len(results) == 0 else 0.0

    if len(results) == 0:
        return 0.0

    # true_results.sort(key=lambda x: x[1])
    # results.sort(key=lambda x: x[1])

    true_indices = set(i for i, _ in true_results)
    indices = set(i for i, _ in results)

    return len(true_indices.intersection(indices)) / len(true_indices)
