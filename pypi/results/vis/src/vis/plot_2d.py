"""Utilities for plotting 2D data."""

import logging
import pathlib

import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def plot_logs(
    inp_path: pathlib.Path,
    out_path: pathlib.Path,
) -> None:
    """Plot logs from the input path."""
    logs: numpy.ndarray = numpy.load(inp_path)
    logger.info(f"Logs shape: {logs.shape}, dtype: {logs.dtype}")

    # There should be three columns in the logs
    if logs.shape[1] != 3:
        msg = f"Unexpected number of columns in the logs: {logs.shape[1]}. Expected 3."
        raise ValueError(msg)

    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot()

    # The columns are kinetic, potential, and total energy
    x = numpy.arange(logs.shape[0])
    ax.scatter(x, logs[:, 0], label="Kinetic Energy")
    ax.scatter(x, logs[:, 1], label="Potential Energy")
    ax.scatter(x, logs[:, 2], label="Total Energy")

    ax.set_xlabel("Time-step")
    ax.set_ylabel("Energy")
    ax.legend()

    plt.savefig(out_path)
    plt.close("all")
