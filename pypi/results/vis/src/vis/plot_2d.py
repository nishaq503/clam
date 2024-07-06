"""Utilities for plotting 2D data."""

import logging
import pathlib
import typing

import fastgif
import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def helper_fastgif(
    inp_dir: pathlib.Path,
) -> typing.Callable[[int], Figure]:
    """A helper function to make a GIF frame for the fastgif package.

    Args:
        inp_dir: Directory containing 2d data from each step in the simulation.

    Returns:
        A function that takes an integer time-step and returns a matplotlib Figure.
    """

    def _make_frame(step: int) -> Figure:
        """Make a frame for the GIF."""
        data: numpy.ndarray = numpy.load(inp_dir / f"{step}.npy")
        fig: Figure = plt.figure()
        ax: Axes = fig.add_subplot()
        ax.scatter(data[:, 0], data[:, 1])
        return fig

    return _make_frame


def plot_gif(
    inp_dir: pathlib.Path,
    out_path: pathlib.Path,
) -> None:
    """Plot a GIF from the input directory."""
    make_frame = helper_fastgif(inp_dir)

    # Get the number of steps in the input directory
    num_steps = len(list(inp_dir.iterdir()))

    fastgif.make_gif(
        fig_fn=make_frame,
        num_calls=num_steps,
        filename=str(out_path),
        num_processes=None,
        show_progress=True,
    )


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
