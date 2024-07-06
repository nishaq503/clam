"""Reduce dimensionality and make plots using `UMAP`."""

import logging
import pathlib

import numpy
import umap

logger = logging.getLogger(__name__)


def umap_reduce(
    path: pathlib.Path,
) -> numpy.ndarray:
    """Reduce the dimensionality of the data using `UMAP`.

    Args:
        path: The path to the data to reduce.

    Returns:
        The reduced data.

    Raises:
        `FileNotFoundError`: If the `path` does not exist.
        `numpy.load`: If the data cannot be loaded.
    """

    if not path.exists():
        msg = f"Path {path} does not exist."
        raise FileNotFoundError(msg)

    data = numpy.load(path)
    reducer = umap.UMAP(
        n_neighbors=10,
        min_dist=0.1,
        n_components=3,
        metric="euclidean",
    )

    return reducer.fit_transform(data)
