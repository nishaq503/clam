"""Plotting various Cluster properties vs depth."""

# pyright: reportUnknownMemberType=false, reportUnknownLambdaType=false

import enum
import pathlib

import pandas
import plotly.express as px
import typer

EXPECTED_COLUMNS = [
    "center_id",
    "depth",
    "cardinality",
    "radius",
    "lfd",
    "radial_sum",
    "span",
    "num_children",
]
"""Expected columns in the csv file containing cluster properties."""

class PlottableColumns(enum.StrEnum):
    """Columns that may be plotted on the y-axis against depth on the x-axis."""

    Cardinality = "cardinality"
    Radius = "radius"
    LFD = "lfd"
    RadialSum = "radial_sum"
    Span = "span"


def plot_cluster_properties(
    data_dir: pathlib.Path = typer.Option(  # noqa: B008
        ...,
        "-d",
        "--data-dir",
        exists=True,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Directory containing the cluster properties csv file.",
    ),
    plots_dir: pathlib.Path = typer.Option(  # noqa: B008
        ...,
        "-p",
        "--plots-dir",
        exists=True,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        help="Directory to save the plots.",
    ),
    dataset: str = typer.Option(
        ...,
        "-D",
        "--dataset",
        help="Name of the dataset used to generate the cluster tree.",
    ),
    strategy: str = typer.Option(
        ...,
        "-s",
        "--strategy",
        help="Clustering strategy used to generate the cluster tree.",
    ),
) -> None:
    """Plot various Cluster properties vs depth."""
    data_dir = data_dir / dataset
    if not data_dir.exists():
        raise FileNotFoundError(f"Could not find {data_dir}")

    tree_csv_path = data_dir / f"{strategy}-tree.csv"
    if not tree_csv_path.exists():
        raise FileNotFoundError(f"Could not find {tree_csv_path}")

    tree_props = pandas.read_csv(tree_csv_path)
    # Check if all expected columns are present
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in tree_props.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in {tree_csv_path}: {missing_columns}")

    # Convert the dtype of the radius, radial_sum, and span columns to float
    tree_props["radius"] = tree_props["radius"].astype(float)
    tree_props["radial_sum"] = tree_props["radial_sum"].astype(float)
    tree_props["span"] = tree_props["span"].astype(float)

    typer.echo(f"Read {len(tree_props)} rows from {tree_csv_path}")
    typer.echo(f"Columns: {tree_props.columns.tolist()}")
    typer.echo(f"Data types:\n{tree_props.dtypes}")

    # Drop all rows where num_children is 0 (i.e., leaf nodes)
    tree_props = tree_props[tree_props["num_children"] > 0]
    for y_axis in PlottableColumns:
        # Drop all columns except depth and the selected y-axis column
        filtered_props = tree_props[["depth", y_axis.value]]
        # Group by depth and compute the min, 5th-percentile, 25th-percentile, median,
        # 75th-percentile, 95th-percentile, and max for each depth group
        filtered_props = filtered_props.groupby("depth").agg(
            min=(y_axis.value, "min"),
            p5=(y_axis.value, lambda x: x.quantile(0.05)),
            p25=(y_axis.value, lambda x: x.quantile(0.25)),
            median=(y_axis.value, "median"),
            p75=(y_axis.value, lambda x: x.quantile(0.75)),
            p95=(y_axis.value, lambda x: x.quantile(0.95)),
            max=(y_axis.value, "max"),
        ).reset_index()

        # Melt the dataframe to have a long format suitable for plotly
        df_melted = filtered_props.melt(
            id_vars="depth",
            var_name="statistic",
            value_name=y_axis.value,
        )
        fig = px.line(
            df_melted,
            x="depth",
            y=y_axis.value,
            color="statistic",
            title=f"{y_axis.value} vs Depth ({strategy})",
            labels={"depth": "Depth", y_axis.value: y_axis.value, "statistic": "Statistic"},
        )
        fig.update_layout(template="plotly_white")
        if y_axis not in [PlottableColumns.LFD]:
            # Use a log scale for y-axis
            fig.update_yaxes(type="log")

        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plots_dir / f"{strategy}-{y_axis.value}-vs-depth.html"
        fig.write_html(plot_path)
        typer.echo(f"Plot saved to {plot_path}")



__all__ = ["plot_cluster_properties"]
