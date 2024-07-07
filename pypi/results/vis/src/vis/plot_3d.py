"""Plotting functions for 3D data."""

import logging
import pathlib
import typing

import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


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

    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot(projection="3d")

    xs = data[:, 0]
    ys = data[:, 1]
    zs = data[:, 2]

    if labels is None:
        ax.scatter(xs, ys, zs)
    else:
        unique_labels = numpy.unique(labels)
        for label in unique_labels:
            mask = labels == label
            ax.scatter(xs[mask], ys[mask], zs[mask], label=label)

    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")
    ax.set_title(path.stem)

    plt.savefig(path)
    plt.close("all")
