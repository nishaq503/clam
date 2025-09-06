"""Plotting various Cluster properties vs depth."""

# pyright: reportUnknownMemberType=false, reportUnknownLambdaType=false

import concurrent.futures
import pathlib
import time

import lmfit
import numpy
import pandas
import plotly.express as px
import plotly.graph_objects as go
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

    # Compute the height of the triangle using Heron's formula
    s = (l_distances + r_distances + base) / 2.0
    area = numpy.sqrt(s * (s - l_distances) * (s - r_distances) * (s - base))
    h = (2.0 * area) / base

    # Compute the projection and preserve the sign using the law of cosines
    cos_l = numpy.square(l_distances) + numpy.square(base) - numpy.square(r_distances)
    projection = numpy.sqrt(numpy.square(l_distances) - numpy.square(h))
    projection *= numpy.sign(cos_l)

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
    annotation: tuple[int, int] | None,
    depth: int,
    cardinality: int,
) -> tuple[go.Scatter, float, int, int] | None:
    """Fit a Gaussian to the projected polar distances in a separate thread.

    Args:
        l_distances: Distances from the left pole.
        r_distances: Distances from the right pole.
        annotation: The annotation tuple containing (span, inter-polar distance).
        depth: The depth of the cluster.
        cardinality: The cardinality of the cluster.

    Returns:
        A tuple containing:
            - trace: The Plotly Scatter trace representing the fitted Gaussian.
            - r2_score: The R² score of the fit.
            - depth: The depth of the cluster.
            - cardinality: The cardinality of the cluster.
    """
    if annotation is None:
        return None

    base = numpy.float64(annotation[1])
    if base < 1e-5:  # noqa: PLR2004
        return None  # Skip if the base distance is too small

    projection = project_distances(l_distances, r_distances, base) / base

    # use numpy to calculate histogram
    counts, bin_edges = numpy.histogram(projection, bins=100)
    midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2.0

    # Calculate the maximum, mean, and standard deviation of the counts
    amplitude_init = numpy.max(counts)
    mean_init = numpy.mean(projection)
    stddev_init = numpy.std(projection)

    if stddev_init < 1e-5:  # noqa: PLR2004
        return None  # Skip if the initial stddev is too small

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

    if fitted_amplitude < 1e-5 or fitted_amplitude > 1e6:  # noqa: PLR2004
        return None  # Skip if the fitted amplitude is too high or too low

    if fitted_stddev < 1e-5:  # noqa: PLR2004
        return None  # Skip if the fitted stddev is too small

    # Calculate R^2 to assess the goodness of fit
    fitted_counts = gaussian(midpoints, fitted_amplitude, fitted_mean, fitted_stddev)
    r2 = r2_score(counts, fitted_counts)

    left, right = fitted_mean - fitted_stddev, fitted_mean + fitted_stddev
    trace = go.Scatter(
        x=[left, fitted_mean, right],
        y=[fitted_amplitude, fitted_amplitude, fitted_amplitude],
        mode="lines",
        name=f"R²={r2:.3f}",
        line=dict(width=1),  # noqa: C408
    )

    return trace, r2, depth, cardinality


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

    min_car, max_car = 10, 2_000_000
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

    traces: list[go.Scatter] = []
    r2_scores: list[float] = []
    depths: list[int] = []
    cardinalities: list[int] = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for cluster in clusters:
            if cluster.annotation is None:
                continue

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
                    cluster.annotation,
                    cluster.depth,
                    cluster.cardinality,
                ),
            )

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Fitting Gaussians"):
            result = future.result()
            if result is None:
                continue
            trace, r2_score, depth, cardinality = result
            traces.append(trace)
            r2_scores.append(r2_score)
            depths.append(depth)
            cardinalities.append(cardinality)

    typer.echo(f"Fitted Gaussians to {len(traces)} clusters")

    # Create a figure with two subplots
    fig1 = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.85, 0.15],
        subplot_titles=("Polar Distances Projections", "R² Histogram"), # pyright: ignore[reportArgumentType]
        vertical_spacing=0.1,
    )

    # Hide the legend
    fig1.update_layout(showlegend=False)

    # Set the x-axis range for both subplots
    fig1.update_xaxes(dict(range=[-0.05, 1.05]), row=1, col=1)  # noqa: C408
    fig1.update_xaxes(dict(range=[-0.05, 1.05]), row=2, col=1)  # noqa: C408

    # Set appropriate titles and labels for the axes
    fig1.update_xaxes(title_text="Projected Distance (normalized)", row=1, col=1)
    fig1.update_xaxes(title_text="R² Score", row=2, col=1)
    fig1.update_yaxes(title_text="Amplitude", type="log", row=1, col=1)
    fig1.update_yaxes(title_text="# Clusters", row=2, col=1)

    # Add traces to the first subplot
    for trace in tqdm(traces, desc="Adding traces to figure"):
        fig1.add_trace(trace, row=1, col=1)

    # Add a histogram of the R^2 scores to the second subplot
    counts, bin_edges = numpy.histogram(r2_scores, bins=100, range=(0.0, 1.0))
    midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2.0
    fig1.add_trace(
        go.Scatter(
            x=midpoints,
            y=counts,
            mode="lines+markers",
            name="R² Histogram",
        ),
        row=2,
        col=1,
    )

    plot_path = plots_dir / "polar_distances.html"
    typer.echo(f"Saving polar distances plot to {plot_path} with html format")
    fig1.write_html(plot_path)

    # Create a pandas DataFrame for easier handling for the second plot
    explorations_df = pandas.DataFrame({
        "R2": r2_scores,
        "Depth": depths,
        "Cardinality": cardinalities,
    })

    # Create a second figure for scatter plots of depth and cardinality vs R²
    fig2 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Depth vs R²", "Cardinality vs R²"), # pyright: ignore[reportArgumentType]
        horizontal_spacing=0.075,
    )
    fig2.update_xaxes(title_text="R² Score", row=1, col=1)
    fig2.update_xaxes(title_text="R² Score", row=1, col=2)
    fig2.update_yaxes(title_text="Depth", row=1, col=1)
    fig2.update_yaxes(title_text="Cardinality", row=1, col=2)

    fig2a = px.scatter(
        explorations_df,
        x="R2",
        y="Depth",
        color="Cardinality",
        color_continuous_scale="bluered",
    )
    fig2a.update_traces(marker=dict(size=3))  # noqa: C408

    fig2b = px.scatter(
        explorations_df,
        x="R2",
        y="Cardinality",
        color="Depth",
        color_continuous_scale="bluered",
    )
    fig2b.update_traces(marker=dict(size=3))  # noqa: C408
    for trace in fig2a.data:
        fig2.add_trace(trace, row=1, col=1)
    for trace in fig2b.data:
        fig2.add_trace(trace, row=1, col=2)

    # Set the y-axis to logarithmic scale for cardinality plot
    fig2.update_yaxes(type="log", row=1, col=2)

    plot_path2 = plots_dir / "r2_scatter.html"
    typer.echo(f"Saving R² scatter plots to {plot_path2} with html format")
    fig2.write_html(plot_path2)

__all__ = ["polar_distances"]
