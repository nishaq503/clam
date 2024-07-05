"""Let's look at some visualizations!"""

import logging
import pathlib
import typing

import numpy
import umap

logger = logging.getLogger("results-vis-init")


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


def scatter_plot(
    data: numpy.ndarray,
    labels: typing.Optional[numpy.ndarray],
    path: pathlib.Path,
) -> None:
    """Create a 3D scatter plot of the data.

    Args:
        data: The data to plot.
        labels: The labels for the data.
        path: The path to save the plot.
    """

    logger.info(f"Data shape: {data.shape}, dtype: {data.dtype}")
    if labels is not None:
        logger.info(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
    logger.info(f"Saving plot to {path}")
