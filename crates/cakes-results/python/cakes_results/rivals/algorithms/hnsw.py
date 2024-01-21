"""HNSW algorithm."""

import logging
import pathlib
import time
import typing

import hnswlib
import numpy

from . import base
from . import utils


class Hnsw(base.Algorithm):
    """Runs the HNSW algorithm."""

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return "hnsw"

    @property
    def has_rnn(self) -> bool:
        """Whether the algorithm supports rnn-search."""
        return False

    @property
    def should_tune(self) -> bool:
        """Whether the algorithm should be tuned."""
        return True

    @property
    def tuned_params(self) -> dict[str, typing.Any]:
        """Parameters tuned for target recall."""
        return {
            "ef_construction": self.ef_construction,
            "M": self.M,
        }

    def __init__(self, *, ef_construction: int = 100, M: int = 16) -> None:
        super().__init__()

        self.index = None
        self.ef_construction = ef_construction
        self.M = M

        self.k = None
        self.results = None

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
        if not 0.0 < target_recall <= 1.0:
            raise ValueError(
                f"Invalid target recall {target_recall}. Must be in (0, 1]."
            )

        efs = [100, 200, 500]
        Ms = [8, 16, 32]

        start = time.perf_counter()
        best_recall, best_ef, best_m = self._tune_index(
            efs,
            Ms,
            ks,
            train,
            metric,
            target_recall,
            test,
            logger,
            gt_dir,
            dataset_name,
            scale,
            max_search_time,
        )
        end = time.perf_counter() - start

        self.ef_construction = best_ef
        self.M = best_m

        logger.info(
            f"Tuned {self.name} in {end:.2e} sec to achieve {best_recall = :.2e}"
            f" at {best_ef = }, {best_m = }."
        )
        return end

    def _tune_index(
        self,
        efs: list[int],
        Ms: list[int],
        ks: list[int],
        train: numpy.ndarray,
        metric: base.Metric,
        target_recall: float,
        test: numpy.ndarray,
        logger: logging.Logger,
        gt_dir: pathlib.Path,
        dataset_name: str,
        scale: int,
        max_search_time: float,
    ) -> tuple[float, int, int]:
        best_recall = 0.0
        best_ef = 0
        best_m = 0

        for ef in efs:
            self.ef_construction = ef

            for m in Ms:
                logger.info(f"Trying {ef = }, {m = } for {self.name}...")
                self.M = m

                self.build_index(train, metric)

                for k in ks:
                    logger.info(f"Trying {k = } for {self.name}...")
                    _, results = self.knn_search(test, k, max_search_time)
                    true_results = utils.load_knn_results(
                        gt_dir, dataset_name, metric, scale, k
                    )

                    recall = utils.compute_recall(true_results, results)
                    if recall > best_recall:
                        best_recall = recall
                        best_ef = ef
                        best_m = m

                    logger.info(f"Achieved {recall = :.2e} at {k = }")
                    if recall < target_recall:
                        break
                else:
                    logger.info(
                        f"Tuned {self.name} to achieve {best_recall = :.2e} at "
                        f"{best_ef = }, {best_m = }."
                    )
                    return best_recall, best_ef, best_m
        else:
            msg = f"Could not tune {self.name} to achieve {target_recall = }."
            logger.error(msg)
            return best_recall, best_ef, best_m

    def build_index(
        self, train: numpy.ndarray, metric: base.Metric
    ) -> tuple[int, int, float]:
        """Build index on the dataset."""

        # Get the cardinality and dimensionality of the dataset
        cardinality, dimensionality = train.shape

        # Build the index
        start = time.perf_counter()
        if metric == base.Metric.Euclidean:
            index = hnswlib.Index(space="l2", dim=dimensionality)
            index.init_index(
                max_elements=cardinality, ef_construction=self.ef_construction, M=self.M
            )
            index.add_items(train)
            self.index = index
        elif metric == base.Metric.Cosine:
            index = hnswlib.Index(space="cosine", dim=dimensionality)
            index.init_index(
                max_elements=cardinality, ef_construction=self.ef_construction, M=self.M
            )
            index.add_items(train)
            self.index = index
        else:
            raise ValueError(f"Invalid metric {metric}.")
        time_taken = time.perf_counter() - start

        return cardinality, dimensionality, time_taken

    def batch_knn_search(
        self, test: numpy.ndarray, k: int
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Run knn-search on the queries."""

        labels, distances = self.index.knn_query(test, k)  # type: ignore[attr-defined]

        return distances, labels

    def batch_rnn_search(
        self, test: numpy.ndarray, radius: float
    ) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Run rnn-search on the queries."""
        raise NotImplementedError("RNN search is not supported by HNSW.")
