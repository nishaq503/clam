"""Parses the SILVA-18S sequences from a FASTA file into a plain text file."""

import collections
import enum
import json
import logging
import pathlib

from fasta_reader import read_fasta
from matplotlib import pyplot as plt
import pandas
import pydantic
import typer

logger = logging.getLogger("parse_silva")
logger.setLevel("INFO")

app = typer.Typer()


@app.command()
def parse(
    fasta_file: pathlib.Path = typer.Option(
        ...,
        "--fasta-file",
        "-i",
        help="The FASTA file containing the SILVA-18S sequences.",
        exists=True,
        readable=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    output_dir: pathlib.Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="The directory where the output files will be saved.",
        exists=True,
        readable=True,
        writable=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    progress_interval: int = typer.Option(
        1_000,
        "--progress-interval",
        "-p",
        help="The number of sequences to parse before logging a progress update.",
    ),
) -> None:
    """Parses the SILVA-18S sequences from a FASTA file.

    The FASTA file is expected to contain the SILVA-18S sequences. The sequences
    are parsed into four files:

    1. A plain text file containing the sequences as they appear in the FASTA
    file. The sequences are separated by newlines.

    2. A plain text file containing the sequences with all gaps removed. The
    sequences are separated by newlines.

    3. A plain text file containing the headers of the sequences. The headers
    are separated by newlines.

    4. A plain text file containing the alphabet and the number of times each
    character appears in the sequences. The first line contains the alphabet
    in sorted order. The remaining lines contain the character and the number
    of times it appears in the sequences.
    """
    logger.info(f"parsing: {fasta_file}")

    stem = fasta_file.stem
    pre_aligned_path = output_dir.joinpath(f"{stem}-pre-aligned.txt")
    unaligned_path = output_dir.joinpath(f"{stem}-unaligned.txt")
    headers_path = output_dir.joinpath(f"{stem}-headers.txt")

    alphabet: dict[str, int] = collections.defaultdict(int)
    num_sequences = 0

    with (
        pre_aligned_path.open("w") as pre_aligned_file,
        unaligned_path.open("w") as unaligned_file,
        headers_path.open("w") as headers_file,
    ):
        for item in read_fasta(str(fasta_file)):
            header = item.defline
            headers_file.write(header + "\n")

            sequence = item.sequence
            pre_aligned_file.write(sequence + "\n")

            sequence = sequence.replace("-", "").replace(".", "")
            unaligned_file.write(sequence + "\n")

            alphabet_counts = collections.Counter(sequence)
            for character, count in alphabet_counts.items():
                alphabet[character] += count

            num_sequences += 1
            if num_sequences % progress_interval == 0:
                logger.info(f"parsed {num_sequences} sequences ...")

    logger.info(f"parsed {num_sequences} sequences")

    alphabet_path = output_dir.joinpath(f"{stem}-alphabet.txt")

    with alphabet_path.open("w") as alphabet_file:
        characters = list(sorted(alphabet.keys()))
        alphabet_file.write("".join(characters) + "\n")

        for character in characters:
            alphabet_file.write(f"{character} {alphabet[character]}\n")


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
    """Plots the search results for the Silva dataset."""
    fig: plt.Figure
    ax: plt.Axes
    fig_scale = 0.8
    fig_size = (8 * fig_scale, 5 * fig_scale)

    # Read the reports
    reports = Report.from_dir(reports_dir)

    # Convert the reports into a DataFrame
    df = reports_to_df(reports)

    for kind, kind_df in df.groupby("kind"):
        assert kind in ["knn", "rnn"]

        if kind == "knn":
            for val, val_df in kind_df.groupby("val"):
                fig, ax = plt.subplots(figsize=fig_size)

                assert val in [10, 100]
                output_path = output_dir.joinpath(f"silva-knn-{val}.png")

                for algorithm, algorithm_df in val_df.groupby("algorithm"):
                    marker = Markers(algorithm)
                    if marker == Markers.Sieve:
                        continue

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
                    title = f"{kind.upper()} {val}"
                    ax.set_title(title)

                ax.set_xlabel("Cardinality")
                ax.set_ylabel("Throughput (queries/s)")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(
                    val_df["cardinality"].min() * 0.75,
                    val_df["cardinality"].max() * 1.25,
                )
                ax.set_ylim(
                    val_df["throughput"].min() * 0.75,
                    val_df["throughput"].max() * 1.25,
                )
                ax.legend(loc="lower left")
                fig.tight_layout()
                fig.savefig(output_path, dpi=300)
                plt.close(fig)

        elif kind == "rnn":
            marker = Markers.Clustered
            fig, ax = plt.subplots(figsize=fig_size)
            output_path = output_dir.joinpath("silva-rnn.png")

            for val, val_df in kind_df.groupby("val"):
                print(f"plotting rnn {val = }")
                assert val in [v * 180 for v in [25, 100, 250]]
                cardinality_throughput = [
                    (cardinality, throughput)
                    for cardinality, throughput in zip(
                        val_df["cardinality"],
                        val_df["throughput"],
                    )
                ]
                cardinality_throughput.sort(key=lambda x: x[0])
                cardinalities, throughput = zip(*cardinality_throughput)

                ax.plot(
                    cardinalities,
                    throughput,
                    label=f"Radius {val // 180}",
                    marker=marker.rnn_marker(val // 180),
                )

            if make_title:
                title = f"{kind.upper()}"
                ax.set_title(title)

            ax.set_xlabel("Cardinality")
            ax.set_ylabel("Throughput (queries/s)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(
                kind_df["cardinality"].min() * 0.75,
                kind_df["cardinality"].max() * 1.25,
            )
            ax.set_ylim(
                kind_df["throughput"].min() * 0.75,
                kind_df["throughput"].max() * 1.25,
            )
            ax.legend(loc="lower left")
            fig.tight_layout()
            fig.savefig(output_path, dpi=300)
            plt.close(fig)

    # Save the dataframe to a csv file
    df.to_csv(output_dir.joinpath("silva-results.csv"), index=False)


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

    def rnn_marker(self, radius: int) -> str:
        """Return the marker for radius in the RNN plot."""
        if radius == 25:
            m = "x"
        elif radius == 100:
            m = "."
        elif radius == 250:
            m = "o"
        else:
            raise ValueError(f"Unknown radius {radius}")
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
        elif self == Markers.Clustered:
            c = "tab:cyan"
        else:
            raise ValueError(f"Unknown algorithm {self}")
        return c


class Report(pydantic.BaseModel):
    """A report for a single dataset."""

    dataset: str
    metric: str
    cardinality: int
    build_time: float
    shard_sizes: list[int]
    num_queries: int
    kind: str
    val: int
    algorithm: str
    throughput: float
    linear_throughput: float
    mean_recall: float

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
