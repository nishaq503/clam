"""Plots for Radio-ML data."""

import enum
import json
import logging
import pathlib

from matplotlib import pyplot as plt
import pandas
import pydantic
import typer

logger = logging.getLogger("radio-ml")
logger.setLevel("INFO")

app = typer.Typer()


@app.command()
def create_plots(
    reports_dir: pathlib.Path = typer.Option(
        ...,
        "--reports-dir",
        "-i",
        help="The directory containing the knn reports.",
        exists=True,
        readable=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    make_title: bool = typer.Option(
        False,
        "--make-title",
        "-t",
        help="Whether to make a title for each plot.",
    ),
    output_dir: pathlib.Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="The directory where the plots will be saved.",
        exists=True,
        readable=True,
        writable=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
) -> None:
    """Plots the search results for the Radio_ML dataset."""

    # Read the reports
    reports = Report.from_dir(reports_dir)

    # Convert the reports into a DataFrame
    df = reports_to_df(reports)

    fig: plt.Figure
    ax: plt.Axes
    fig_scale = 0.8
    fig_size = (8 * fig_scale, 5 * fig_scale)

    for k, k_df in df.groupby("k"):
        fig, ax = plt.subplots(figsize=fig_size)

        assert k in [10, 100]
        output_path = output_dir.joinpath(f"radio-ml-knn-{k}.png")

        for algorithm, algorithm_df in k_df.groupby("algorithm"):
            if algorithm == "Sieve":
                continue

            marker = Markers(algorithm)
            cardinality_throughput = [
                (cardinality, throughput)
                for cardinality, throughput in zip(
                    algorithm_df["cardinality"],
                    algorithm_df["throughput"],
                )
            ]
            cardinality_throughput.sort(key=lambda x: x[0])
            cardinalities, throughput = zip(*cardinality_throughput)

            ax.plot(
                cardinalities,
                throughput,
                label=marker.alias(),
                marker=marker.marker(),
                color=marker.color(),
            )

        if make_title:
            title = f"knn {k}"
            ax.set_title(title)

        ax.set_xlabel("Cardinality")
        ax.set_ylabel("Throughput (queries/s)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(
            k_df["cardinality"].min() * 0.75,
            k_df["cardinality"].max() * 1.25,
        )
        ax.set_ylim(
            k_df["throughput"].min() * 0.75,
            k_df["throughput"].max() * 1.25,
        )
        ax.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    # # Save the dataframe to a csv file
    # df.to_csv(output_dir.joinpath("radio-ml-results.csv"), index=False)


class Markers(str, enum.Enum):
    GreedySieve = "GreedySieve"
    Linear = "Linear"
    RepeatedRnn = "RepeatedRnn"
    Sieve = "Sieve"
    SieveSepCenter = "SieveSepCenter"
    Clustered = "Clustered"

    def marker(self) -> str:
        """Return the marker for the algorithm."""
        if self == Markers.GreedySieve:
            m = "x"
        elif self == Markers.Linear:
            m = "."
        elif self == Markers.RepeatedRnn:
            m = "o"
        elif self == Markers.Sieve:
            m = "s"
        elif self == Markers.SieveSepCenter:
            m = "d"
        elif self == Markers.Clustered:
            m = ">"
        else:
            raise ValueError(f"Unknown algorithm {self}")
        return m

    def alias(self) -> str:
        """Return the alias for the algorithm."""
        if self == Markers.GreedySieve:
            a = "DepthFirstSieve"
        elif self == Markers.Linear:
            a = "Linear"
        elif self == Markers.RepeatedRnn:
            a = "RepeatedRNN"
        elif self == Markers.Sieve:
            a = "SieveBuggy"
        elif self == Markers.SieveSepCenter:
            a = "BreadthFirstSieve"
        elif self == Markers.Clustered:
            a = "Clustered"
        else:
            raise ValueError(f"Unknown algorithm {self}")
        return f"Cakes-{a}"

    def color(self) -> str:
        """Consistent color for the algorithm across plots."""
        if self == Markers.GreedySieve:
            c = "tab:blue"
        elif self == Markers.Linear:
            c = "tab:orange"
        elif self == Markers.RepeatedRnn:
            c = "tab:green"
        elif self == Markers.Sieve:
            c = "tab:red"
        elif self == Markers.SieveSepCenter:
            c = "tab:purple"
        else:
            raise ValueError(f"Unknown algorithm {self}")
        return c


class Report(pydantic.BaseModel):
    """A report for a single dataset."""

    dataset: str
    metric: str
    cardinality: int
    built: bool
    build_time: float
    shard_sizes: list[int]
    num_queries: int
    k: int
    algorithm: str
    throughput: float

    @staticmethod
    def from_path(path: pathlib.Path) -> "Report":
        """Reads a report from a file."""
        with path.open("r") as file:
            return Report(**json.load(file))

    @staticmethod
    def from_dir(dir: pathlib.Path) -> list["Report"]:
        """Reads all reports from a directory."""
        return [Report.from_path(path) for path in dir.glob("*.json")]


def reports_to_df(
    reports: list[Report],
) -> pandas.DataFrame:
    """Converts a list of reports into a DataFrame."""

    # Convert the reports into a list of dictionaries.
    data = [
        report.model_dump(exclude=["shard_sizes", "mean_recall"]) for report in reports
    ]

    # Convert the list of dictionaries into a DataFrame.
    return pandas.DataFrame(data)
