"""FAISS IndexIVFFlat algorithm."""

import logging
import pathlib
import time
import typing

import faiss
import numpy

from . import base
from . import faiss_flat
from . import utils


class FaissIVFFlat(faiss_flat.FaissFlat):
    """Runs the FAISS IndexFlatIVF algorithm."""

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return "faiss-ivf-flat"

    def __init__(self, *, nlist: int = 100, nprobe: int = 100) -> None:
        super().__init__()

        self.index = None
        self.nlist = nlist
        self.nprobe = nprobe

    @property
    def should_tune(self) -> bool:
        """Whether the algorithm should be tuned."""
        return True

    @property
    def tuned_params(self) -> dict[str, typing.Any]:
        """Parameters tuned for target recall."""
        return {
            "nlist": self.nlist,
            "nprobe": self.nprobe,
        }

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
            max_search_time: Maximum time to spend searching.

        Returns:
            time_taken: Time taken to tune the index.
        """
        if not 0.0 < target_recall <= 1.0:
            raise ValueError(
                f"Invalid target recall {target_recall}. Must be in (0, 1]."
            )

        # Get the cardinality and dimensionality of the dataset
        cardinality, _ = train.shape

        params = [10**i for i in range(0, 3) if 10**i < cardinality / 10]

        # Build the index
        start = time.perf_counter()
        for nlist in reversed(params):
            self.nlist = nlist

            logger.info(f"Trying {nlist = } for {self.name}...")
            self.build_index(train, metric)

            for nprobe in params:
                if nprobe > nlist:
                    continue

                logger.info(f"Trying {nlist = }, {nprobe = } for {self.name}...")
                self.nprobe = nprobe

                for k in ks:
                    _, knn_results = self.knn_search(test, k, max_search_time)
                    true_results = utils.load_knn_results(
                        gt_dir, dataset_name, metric, scale, k
                    )
                    recall = utils.compute_recall(true_results, knn_results)
                    logger.info(f"Achieved {recall = :.2e} for {k = }")
                    if recall < target_recall:
                        break
                else:
                    for radius in radii:
                        _, rnn_results = self.rnn_search(test, radius, max_search_time)
                        true_results = utils.load_rnn_results(
                            gt_dir, dataset_name, metric, scale, radius
                        )
                        recall = utils.compute_recall(true_results, rnn_results)
                        logger.info(f"Achieved {recall = :.2e} for {radius = }")
                        if recall < target_recall:
                            break
                    else:
                        end = time.perf_counter() - start
                        logger.info(
                            f"Tuned {self.name} in {end:.2f}s with {nlist = }, {nprobe = }."  # noqa: E501
                        )
                        return end
        else:
            end = time.perf_counter() - start
            msg = f"Could not tune {self.name} to achieve {target_recall = }."
            logger.error(msg)
            return end

    def build_index(
        self, train: numpy.ndarray, metric: base.Metric
    ) -> tuple[int, int, float]:
        """Build index on the dataset."""

        # Get the cardinality and dimensionality of the dataset
        cardinality, dimensionality = train.shape

        # Build the index
        start = time.perf_counter()
        if metric == base.Metric.Euclidean:
            quantizer = faiss.IndexFlatL2(dimensionality)
            self.index = faiss.IndexIVFFlat(quantizer, dimensionality, self.nlist)
            self.index.train(train)  # type: ignore[attr-defined]
            self.index.add(train)  # type: ignore[attr-defined]
        elif metric == base.Metric.Cosine:
            quantizer = faiss.IndexFlatIP(dimensionality)
            self.index = faiss.IndexIVFFlat(quantizer, dimensionality, self.nlist)
            faiss.normalize_L2(train)
            self.index.train(train)  # type: ignore[attr-defined]
            self.index.add(train)  # type: ignore[attr-defined]
        else:
            raise ValueError(f"Invalid metric {metric}.")
        time_taken = time.perf_counter() - start

        return cardinality, dimensionality, time_taken
