"""ANNOY search algorithm."""

import logging
import pathlib
import time
import typing
import annoy

import numpy

from . import base
from . import utils


class Annoy(base.Algorithm):
    """Runs the ANNOY algorithm."""

    def __init__(self, *, n_trees: int = 10) -> None:
        super().__init__()

        self.index: annoy.AnnoyIndex = None

        self.n_trees = n_trees

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return "annoy"

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
        return {"n_trees": self.n_trees}

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

        n_tree_choices = [10, 20, 50, 100]

        start = time.perf_counter()
        best_recall, best_n_trees = self._tune_index(
            n_tree_choices,
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
        tuning_time = time.perf_counter() - start

        self.n_trees = best_n_trees

        logger.info(
            f"Tuned {self.name} to achieve {best_recall = :.2e} with {best_n_trees = }."
        )
        return tuning_time

    def _tune_index(
        self,
        n_tree_choices: list[int],
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
    ) -> tuple[float, int]:
        best_recall = 0.0
        best_n_trees = n_tree_choices[0]

        for n_trees in n_tree_choices:
            self.n_trees = n_trees

            logger.info(f"Building index with {n_trees = }.")
            self.build_index(train, metric)

            for k in ks:
                logger.info(f"Running knn-search with {k = }.")

                _, results = self.knn_search(test, k, max_search_time)
                true_results = utils.load_knn_results(
                    gt_dir, dataset_name, metric, scale, k
                )

                recall = utils.compute_recall(true_results, results)
                if recall > best_recall:
                    best_recall = recall
                    best_n_trees = n_trees

                logger.info(f"Achieved {recall = :.2e} with {k = }.")
                if recall < target_recall:
                    break
            else:
                logger.info(
                    f"Target recall {target_recall} achieved with {n_trees = }."
                )
                return best_recall, best_n_trees
        else:
            msg = f"Could not tune {self.name} to achieve {target_recall = }."
            logger.error(msg)
            return best_recall, best_n_trees

    def build_index(
        self, train: numpy.ndarray, metric: base.Metric
    ) -> tuple[int, int, float]:
        """Build index on a dataset."""

        cardinality, dimensionality = train.shape

        if metric == base.Metric.Euclidean:
            self.index = annoy.AnnoyIndex(dimensionality, "euclidean")
        elif metric == base.Metric.Cosine:
            self.index = annoy.AnnoyIndex(dimensionality, "angular")
        else:
            raise ValueError(f"Invalid metric {metric}.")

        build_start = time.perf_counter()
        for i, vec in enumerate(train):
            self.index.add_item(i, vec)
        self.index.build(self.n_trees)
        build_time = time.perf_counter() - build_start

        return cardinality, dimensionality, build_time

    def batch_knn_search(
        self, test: numpy.ndarray, k: int
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Run knn-search on the queries."""

        results = [self.index.get_nns_by_vector(vec, k) for vec in test]

        # turn `results` into a 2d numpy array
        neighbors = numpy.array(results)
        assert neighbors.shape == (test.shape[0], k)

        return numpy.zeros_like(neighbors), neighbors

    def batch_rnn_search(
        self, test: numpy.ndarray, radius: float
    ) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Run rnn-search on the queries."""
        raise NotImplementedError("ANNOY does not support rnn-search.")
