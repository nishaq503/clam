"""Base class for running rival search algorithms."""

import abc
import enum
import logging
import pathlib
import time
import typing

import numpy

logger = logging.getLogger(__file__)
logger.setLevel("INFO")


class Metric(str, enum.Enum):
    Euclidean = "euclidean"
    Cosine = "cosine"


class Algorithm(abc.ABC):
    """Base class for a rival search algorithm."""

    @abc.abstractproperty
    def name(self) -> str:
        """Name of the algorithm."""

    @abc.abstractproperty
    def has_rnn(self) -> bool:
        """Whether the algorithm supports rnn-search."""

    @abc.abstractproperty
    def should_tune(self) -> bool:
        """Whether the algorithm should be tuned."""

    @abc.abstractproperty
    def tuned_params(self) -> dict[str, typing.Any]:
        """Parameters tuned for target recall."""

    @abc.abstractmethod
    def tune_index(
        self,
        train: numpy.ndarray,
        metric: Metric,
        target_recall: float,
        test: numpy.ndarray,
        ks: list[int],
        radii: list[float],
        logger: logging.Logger,
        gt_dir: pathlib.Path,
        dataset_name: str,
        scale: int,
        max_search_time: float,
    ) -> float:
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
            gt_dir: Path to the ground truth directory.
            dataset_name: Name of the dataset.
            scale: Scale of the dataset.
            max_search_time: Maximum time to run search for.

        Returns:
            tuning_time: Time taken to tune the index.
        """

    @abc.abstractmethod
    def build_index(
        self, train: numpy.ndarray, metric: Metric
    ) -> tuple[int, int, float]:
        """Build index on a dataset.

        Args:
            train: 2d numpy array of the dataset to build the index on.
            metric: Distance metric to use. Either "euclidean" or "cosine".

        Returns:
            cardinality: Cardinality of the dataset.
            dimensionality: Dimensionality of the dataset.
            index_build_time: Time taken to build the index.
        """

    @abc.abstractmethod
    def batch_knn_search(
        self, test: numpy.ndarray, k: int
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Run knn-search on the queries.

        Args:
            test: n queries to run knn-search on.
            k: Number of nearest neighbors to find.

        Returns:
            distances: n*k array of distances of the nearest neighbors of each
            query.
            neighbors: n*k array of indices of the nearest neighbors of each
            query.
        """

    @abc.abstractmethod
    def batch_rnn_search(
        self, test: numpy.ndarray, radius: float
    ) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Run rnn-search on the queries.

        Args:
            test: n queries to run rnn-search on.
            radius: Radius of the search.

        Returns:
            lims: n+1 array of the limits for each query into the distances and
            neighbors arrays.
            distances: 1d array of distances of the neighbors of each query.
            neighbors: 1d array of indices of the neighbors of each query.
        """

    def knn_search(
        self, test: numpy.ndarray, k: int, max_time: float
    ) -> tuple[float, list[list[tuple[int, float]]]]:
        """Run knn-search on the queries.

        Args:
            test: Queries to run knn-search on.
            k: Number of nearest neighbors to find.
            max_time: Maximum time to run knn-search for.

        Returns:
            throughput: Throughput (QPS) of the search.
            results: List of lists of tuples of (index, distance) of the
            nearest neighbors of each query.
        """

        distances_list, neighbors_list = [], []

        # Get the number of queries
        num_queries = test.shape[0]
        batch_size = 100
        time_taken = 0.0

        # Run knn-search in batches of 100 queries
        for i in range(0, num_queries, batch_size):
            start = time.perf_counter()
            dist, neigh = self.batch_knn_search(test[i : i + batch_size], k)
            time_taken += time.perf_counter() - start

            distances_list.append(dist)
            neighbors_list.append(neigh)

            # Stop if the total time taken exceeds max_time
            if time_taken > max_time:
                break

        results = []
        for dist, neigh in zip(distances_list, neighbors_list):
            batch = [
                [(int(i), float(d)) for i, d in zip(neigh[row], dist[row])]
                for row in range(batch_size)
            ]
            results.extend(batch)

        # Calculate throughput
        throughput = len(results) / time_taken

        return throughput, results

    def rnn_search(
        self, test: numpy.ndarray, radius: float, max_time: float
    ) -> tuple[float, list[list[tuple[int, float]]]]:
        """Run rnn-search on the queries.

        Args:
            test: Queries to run rnn-search on.
            radius: Radius of the search.
            max_time: Maximum time to run rnn-search for.

        Returns:
            throughput: Throughput (QPS) of the search.
            results: List of lists of tuples of (index, distance) of the
            neighbors of each query.
        """

        lims_list, distances_list, neighbors_list = [], [], []

        # Get the number of queries
        num_queries = test.shape[0]
        batch_size = 100
        time_taken = 0.0

        # Run rnn-search in batches of 100 queries
        for i in range(0, num_queries, batch_size):
            start = time.perf_counter()
            lims, dist, neigh = self.batch_rnn_search(test[i : i + batch_size], radius)
            time_taken += time.perf_counter() - start

            lims_list.append(lims)
            distances_list.append(dist)
            neighbors_list.append(neigh)

            # Stop if the total time taken exceeds max_time
            if time_taken > max_time:
                break

        results = []
        for lims, dist, neigh in zip(lims_list, distances_list, neighbors_list):
            batch = [
                [
                    (int(i), float(d))
                    for i, d in zip(
                        neigh[lims[row] : lims[row + 1]],
                        dist[lims[row] : lims[row + 1]],
                    )
                ]
                for row in range(batch_size)
            ]
            results.extend(batch)

        # Calculate throughput
        throughput = len(results) / time_taken

        return throughput, results
