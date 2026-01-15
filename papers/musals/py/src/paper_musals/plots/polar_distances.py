"""Plotting various Cluster properties vs depth."""

# pyright: reportUnknownMemberType=false, reportUnknownLambdaType=false

import pathlib
import time

import numpy
import plotly.graph_objects as go
import plotly.io as pio
import pydantic
import typer

from paper_musals.models import Cluster


def read_clusters[T, A](explorations_dir: pathlib.Path) -> list[Cluster[T, A]]: # pyright: ignore[reportInvalidTypeVarUse]
    """Read list of clusters from JSON file."""
    json_path = explorations_dir / "clusters.json"
    with json_path.open("r") as f:
        contents: str = f.read()
    return pydantic.TypeAdapter(list[Cluster[T, A]]).validate_json(contents)


def read_polar_distances(explorations_dir: pathlib.Path) -> dict[str, numpy.ndarray]:
    """Read polar distances from NPZ file."""
    npz_path = explorations_dir / "polar_distances.npz"

    with numpy.load(npz_path) as data:
        arrays: dict[str, numpy.ndarray] = dict(data)
    return arrays


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


def polar_distances(
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
    clusters: list[Cluster[int, int]] = read_clusters(explorations_dir)
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

    fig = go.Figure()
    min_car, max_car = 10_000, 200_000
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

    missing = []
    projections = []
    for cluster in clusters:
        base = numpy.float64(cluster.radius * 2)
        l_name = f"{cluster.center_index}_l"
        r_name = f"{cluster.center_index}_r"

        if l_name not in arrays or r_name not in arrays:
            if l_name not in arrays:
                typer.echo(f"Warning: missing left polar distances for cluster {cluster.center_index}")
            if r_name not in arrays:
                typer.echo(f"Warning: missing right polar distances for cluster {cluster.center_index}")
            missing.append(cluster.center_index)
            continue

        l_distances, r_distances = arrays[l_name], arrays[r_name]
        projection = project_distances(l_distances, r_distances, base) / base
        fig.add_trace(go.Histogram(x=projection, opacity=0.25, nbinsx=50))
        projections.extend(projection.tolist())
        typer.echo(f"Added projections for cluster {cluster.center_index} with cardinality {cluster.cardinality}")

    typer.echo(f"Computed {len(projections)} projections from polar distances")

    plot_path = plots_dir / "polar_distances.png"
    typer.echo(f"Saving polar distances plot to {plot_path}")
    pio.write_image(fig, plot_path, height=500, width=800, scale=4)
    fig.write_html(plot_path.with_suffix(".html"))

    fig.update_yaxes(type="log")
    log_plot_path = plots_dir / "log_polar_distances.png"
    typer.echo(f"Saving log polar distances plot to {log_plot_path}")
    pio.write_image(fig, log_plot_path, height=500, width=800, scale=4)
    fig.write_html(log_plot_path.with_suffix(".html"))

    if missing:
        typer.echo(f"Total missing clusters: {len(missing)} out of {len(clusters)}")
    else:
        typer.echo("All clusters have polar distances.")

    typer.echo(f"Plots will be saved to {plots_dir}")


__all__ = ["polar_distances"]
