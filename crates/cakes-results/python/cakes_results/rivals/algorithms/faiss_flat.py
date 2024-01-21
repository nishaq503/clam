"""FAISS IndexFlat algorithm."""

import logging
import pathlib
import time
import typing

import faiss
import numpy

from . import base


class FaissFlat(base.Algorithm):
    """Runs the FAISS FlatIndex algorithm."""

    def __init__(self) -> None:
        super().__init__()

        self.index = None

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return "faiss-flat"

    @property
    def has_rnn(self) -> bool:
        """Whether the algorithm supports rnn-search."""
        return True

    @property
    def should_tune(self) -> bool:
        """Whether the algorithm should be tuned."""
        return False

    @property
    def tuned_params(self) -> dict[str, typing.Any]:
        """Parameters tuned for target recall."""
        return {}

    def tune_index(
        self,
        train: numpy.ndarray,
        metric: base.Metric,
        target_recall: float,
        test: numpy.ndarray,
        ks: list[int],
        radii: list[float],
        logger: logging.Logger,
        gt_dir: pathlib.Path,
        dataset_name: str,
        scale: int,
        max_search_time: float,
    ):
        """Builds the index with parameters tuned on the queries to achieve
        the target recall.

        Args:
            train: 2d numpy array of the dataset to build the index on.
            metric: Distance metric to use. Either "euclidean" or "cosine".
            target_recall: Target recall to tune for.
            test: Queries to run knn-search on.
            ks: List of k values to tune for.
            radii: List of radii values to tune for.
            logger: Logger to log to.
            gt_dir: Ground truth directory.
            dataset_name: Name of the dataset.
            scale: Scale of the dataset.
            max_search_time: Maximum time to spend on tuning.

        Returns:
            tuning_time: Time taken to tune the index.
        """

        raise NotImplementedError(
            "Tuning is not supported for the FAISS FlatIndex algorithm."
        )

    def build_index(
        self, train: numpy.ndarray, metric: base.Metric
    ) -> tuple[int, int, float]:
        """Build index on the dataset."""

        # Get the cardinality and dimensionality of the dataset
        cardinality, dimensionality = train.shape

        # Build the index
        start = time.perf_counter()
        if metric == base.Metric.Euclidean:
            self.index = faiss.IndexFlatL2(dimensionality)
            self.index.add(train)  # type: ignore[attr-defined]
        elif metric == base.Metric.Cosine:
            self.index = faiss.IndexFlatIP(dimensionality)
            faiss.normalize_L2(train)
            self.index.add(train)  # type: ignore[attr-defined]
        else:
            raise ValueError(f"Invalid metric {metric}.")
        time_taken = time.perf_counter() - start

        return cardinality, dimensionality, time_taken

    def batch_knn_search(
        self, test: numpy.ndarray, k: int
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Run knn-search on the queries."""

        return self.index.search(test, k)  # type: ignore[attr-defined]

    def batch_rnn_search(
        self, test: numpy.ndarray, radius: float
    ) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Run rnn-search on the queries."""

        return self.index.range_search(test, radius)  # type: ignore[attr-defined]
