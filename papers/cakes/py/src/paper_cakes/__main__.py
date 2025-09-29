"""Making plots for the CAKES paper and comparing to other algorithms."""

import pathlib

import typer

app = typer.Typer()

@app.command()
def main(
    inp_dir: pathlib.Path = typer.Option(  # noqa: B008
        ...,
        "-i",
        help="Input directory containing datasets.",
        dir_okay=True,
        file_okay=False,
        exists=True,
        readable=True,
        resolve_path=True,
    ),
    out_dir: pathlib.Path = typer.Option(  # noqa: B008
        ...,
        "-o",
        help="Output directory to save results.",
        dir_okay=True,
        file_okay=False,
        writable=True,
        resolve_path=True,
    ),
    dataset: str = typer.Option(
        ...,
        "-d",
        help="Name of the dataset file (e.g., 'musk').",
    ),
    metric: str = typer.Option(
        ...,
        "-m",
        help="Distance metric to use (e.g., 'euclidean', 'cosine').",
    ),
) -> None:
    """Reproducible results for the CAKES paper."""
    typer.echo("Running paper-cakes with the following parameters:")
    typer.echo(f"  Input Directory: {inp_dir}")
    typer.echo(f"  Output Directory: {out_dir}")
    typer.echo(f"  Dataset: {dataset}")
    typer.echo(f"  Metric: {metric}")

    # Add the core logic of the py-mbed experiment here.
    err = "Experiment logic not yet implemented."
    raise NotImplementedError(err)


if __name__ == "__main__":
    app()
