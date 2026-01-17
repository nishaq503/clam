"""Plotting various Cluster properties vs depth."""

# pyright: reportUnknownMemberType=false, reportUnknownLambdaType=false

import concurrent.futures
import pathlib
import time

import lmfit
import numpy
import plotly.graph_objects as go
import plotly.io as pio
import pydantic
import typer
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score
from tqdm import tqdm

from paper_musals.models import Cluster


def gaussian(x: numpy.ndarray, amplitude: float, mean: float, stddev: float) -> numpy.ndarray:
    """Gaussian function."""
    stddev += 1e-5  # Prevent division by zero
    return amplitude * numpy.exp(-((x - mean) ** 2) / (2 * stddev ** 2))


def read_clusters[T, A](explorations_dir: pathlib.Path) -> list[Cluster[T, A]]: # pyright: ignore[reportInvalidTypeVarUse]
    """Read list of clusters from JSON file."""
    json_path = explorations_dir / "clusters.json"
    with json_path.open("r") as f:
        contents: str = f.read()
    return pydantic.TypeAdapter(list[Cluster[T, A]]).validate_json(contents)


def project_distances(l_distances: numpy.ndarray, r_distances: numpy.ndarray, base: numpy.float64) -> numpy.ndarray:
    """Project left and right polar distances to a single distance on the line joining the two poles.

    Args:
        l_distances: Distances from the left pole.
        r_distances: Distances from the right pole.
        base: Distance between the two poles.
    """
    l_distances = l_distances.astype(numpy.float64)
    r_distances = r_distances.astype(numpy.float64)
    projection = numpy.zeros_like(l_distances, dtype=numpy.float64)
    for i, (left, right) in enumerate(zip(l_distances, r_distances, strict=True)):
        d: numpy.float64
        if left > base and right > base:
            d = project_triangle(left, right, base)
        elif left > base:
            d = -project_triangle(left, right, base)
        elif right > base:
            d = project_triangle(left, right, base)
        else:
            d = project_triangle(left, right, base)
        projection[i] = d
    return projection


def project_triangle(left: numpy.float64, right: numpy.float64, base: float) -> numpy.float64:
    """Project left and right polar distances to a single distance on the line joining the two poles.

    This assumes that the distances form a valid triangle and that the third point is above and in the middle of the two poles.

    Args:
        left: Distance from the left pole.
        right: Distance from the right pole.
        base: Distance between the two poles.
    """
    # Compute the height of the triangle using Heron's formula
    s = (left + right + base) / 2.0
    area = numpy.sqrt(s * (s - left) * (s - right) * (s - base))
    h = (2.0 * area) / base
    return numpy.sqrt(left**2 - h**2)


def fit_gaussian_thread(
    l_distances: numpy.ndarray,
    r_distances: numpy.ndarray,
    base: numpy.float64,
) -> tuple[float, float, float, float]:
    """Fit a Gaussian to the projected polar distances in a separate thread.

    Args:
        l_distances: Distances from the left pole.
        r_distances: Distances from the right pole.
        base: Distance between the two poles.

    Returns:
        A tuple containing:
            - fitted_amplitude: The amplitude of the fitted Gaussian.
            - fitted_mean: The mean of the fitted Gaussian.
            - fitted_stddev: The standard deviation of the fitted Gaussian.
            - r2: The R² score of the fit.
    """
    projection = project_distances(l_distances, r_distances, base) / base

    # use numpy to calculate histogram
    counts, bin_edges = numpy.histogram(projection, bins=100)
    midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2.0

    # Calculate the maximum, mean, and standard deviation of the counts
    amplitude_init = numpy.max(counts)
    mean_init = numpy.mean(projection)
    stddev_init = numpy.std(projection)

    # Define the Gaussian model using lmfit
    model = lmfit.Model(gaussian)
    params = model.make_params(
        amplitude=amplitude_init,
        mean=mean_init,
        stddev=stddev_init,
    )

    # Perform the fit
    result = model.fit(counts, params, x=midpoints)
    fitted_amplitude = result.params["amplitude"].value
    fitted_mean = result.params["mean"].value
    fitted_stddev = result.params["stddev"].value

    # Calculate R^2 to assess the goodness of fit
    fitted_counts = gaussian(midpoints, fitted_amplitude, fitted_mean, fitted_stddev)
    r2 = r2_score(counts, fitted_counts)

    return fitted_amplitude, fitted_mean, fitted_stddev, r2


def polar_distances(  # noqa: PLR0915
    explorations_dir: pathlib.Path = typer.Option(  # noqa: B008
        ...,
        "-i",
        "--explorations-dir",
        exists=True,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Directory containing the results of the explorations from Rust.",
    ),
    plots_dir: pathlib.Path = typer.Option(  # noqa: B008
        ...,
        "-o",
        "--plots-dir",
        exists=True,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        help="Directory to save the plots.",
    ),
) -> None:
    """Plot the polar distances."""
    start = time.perf_counter()
    clusters: list[Cluster[int, tuple[int, int]]] = read_clusters(explorations_dir)
    end = time.perf_counter() - start
    typer.echo(f"Read {len(clusters)} clusters from {explorations_dir} in {end:.2f} seconds")

    start = time.perf_counter()
    clusters = sorted(clusters, key=lambda c: c.center_index)
    end = time.perf_counter() - start

    npz_path = explorations_dir / "polar_distances.npz"
    start = time.perf_counter()
    with numpy.load(npz_path) as data:
        arrays: dict[str, numpy.ndarray] = dict(data)
    end = time.perf_counter() - start
    typer.echo(f"Read polar distances from {npz_path} in {end:.2f} seconds")

    # Create a figure with two subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.85, 0.15],
        subplot_titles=("Polar Distances Projections", "R² Histogram"), # pyright: ignore[reportArgumentType]
        vertical_spacing=0.1,
    )

    min_car, max_car = 10, 20_000_000
    clusters = [
        c for c in clusters if all(
            [
                c.cardinality > min_car,
                c.cardinality < max_car,
                c.annotation is not None,
            ],
        )
    ]
    typer.echo(f"Filtered clusters to {len(clusters)} with cardinality in ({min_car}, {max_car})")

    params_results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for cluster in clusters:
            if cluster.annotation is None:
                continue

            base = numpy.float64(cluster.annotation[1])
            if base < 1e-5:  # noqa: PLR2004
                continue  # Skip clusters with zero base distance

            l_name = f"{cluster.center_index}_l"
            r_name = f"{cluster.center_index}_r"

            if l_name not in arrays or r_name not in arrays:
                continue

            l_distances, r_distances = arrays[l_name], arrays[r_name]
            futures.append(
                executor.submit(
                    fit_gaussian_thread,
                    l_distances,
                    r_distances,
                    base,
                ),
            )

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Fitting Gaussians"):
            params_results.append(future.result())  # noqa: PERF401

    r2_scores = []
    amplitudes = []
    for params in params_results:
        fitted_amplitude, fitted_mean, fitted_stddev, r2 = params
        # If the fitted amplitude is too high, skip it
        if numpy.log10(fitted_amplitude) > 20:  # noqa: PLR2004
            continue

        amplitudes.append(fitted_amplitude)
        r2_scores.append(r2)

        # Add a horizontal line to represent one standard deviation from the mean
        left, right = fitted_mean - fitted_stddev, fitted_mean + fitted_stddev
        fig.add_trace(
            go.Scatter(
                x=[left, fitted_mean, right],
                y=[fitted_amplitude, fitted_amplitude, fitted_amplitude],
                mode="lines",
                name=f"R²={r2:.3f}",
            ),
            row=1,
            col=1,
        )

    # Make the lines thinner for better visibility, and set the x-axis range to (0, 1)
    fig.update_traces(line=dict(width=1), row=1, col=1)  # noqa: C408
    fig.update_layout(xaxis=dict(range=[-0.1, 1.1]))  # noqa: C408

    # Set the y-axis range
    amp_percentile = numpy.percentile(amplitudes, 99)  # 99th percentile
    fig.update_layout(yaxis=dict(range=[1, amp_percentile * 2.0]))  # noqa: C408

    # Add a histogram of the R^2 scores to the second subplot
    counts, bin_edges = numpy.histogram(r2_scores)
    midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2.0
    fig.add_trace(
        go.Scatter(
            x=midpoints,
            y=counts,
            mode="lines+markers",
            name="R² Histogram",
        ),
        row=2,
        col=1,
    )

    # Set appropriate titles and labels for the axes
    fig.update_xaxes(title_text="Projected Distance (normalized)", row=1, col=1)
    fig.update_yaxes(title_text="Counts", row=1, col=1)
    fig.update_xaxes(title_text="R² Score", row=2, col=1)
    fig.update_yaxes(title_text="# Clusters", row=2, col=1)

    # Hide the legend if there are too many entries
    if len(fig.data) > 10:  # noqa: PLR2004
        typer.echo("Hiding legend due to too many entries")
        fig.update_layout(showlegend=False)

    plot_path = plots_dir / "polar_distances"
    typer.echo(f"Saving polar distances plot to {plot_path} with html and png formats")
    pio.write_image(fig, plot_path.with_suffix(".png"), height=1000, width=1600, scale=4)
    fig.write_html(plot_path.with_suffix(".html"))


__all__ = ["polar_distances"]
