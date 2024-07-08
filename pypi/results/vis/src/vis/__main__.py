"""CLI for the package."""

import logging
import pathlib
import shutil
import sys

import numpy
import typer

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
import vis

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("results-vis")
logger.setLevel("INFO")

app = typer.Typer()


@app.command()
def main(
    inp_dir: pathlib.Path = typer.Option(  # noqa: B008
        ...,
        help="Input directory containing the input data.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    red_dir: pathlib.Path = typer.Option(  # noqa: B008
        ...,
        help="Input directory to save the reduced data from CLAM.",
        exists=False,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    out_dir: pathlib.Path = typer.Option(  # noqa: B008
        ...,
        help="Output directory to save the results.",
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
    ),
) -> None:
    """CLI entry point."""

    logger.info(f"Input directory: {inp_dir}")
    logger.info(f"Reduced directory: {red_dir}")
    logger.info(f"Output directory: {out_dir}")

    datasets = [
        "cardio",
        "arrhythmia",
        "satellite",
        "mnist",  # Sometimes produces infinite forces
    ]
    members = ["cc", "gn", "pc", "sc", "vd"]
    ml_models = ["lr", "dt"]
    for name in datasets:
        # The files should be named as `name.npy` and `name_labels.npy`
        final_path = inp_dir / f"{name}.npy"
        labels_path = inp_dir / f"{name}_labels.npy"
        labels: numpy.ndarray = numpy.load(labels_path)
        logger.info(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")

        # Create a directory for the data
        data_out_dir = out_dir / name
        if data_out_dir.exists():
            shutil.rmtree(data_out_dir)
        data_out_dir.mkdir()

        # Create the dim-reduction for the data and save a scatter plot
        reduced_data = vis.plot_umap.umap_reduce(final_path)
        vis.plot_3d.scatter_plot(reduced_data, labels, data_out_dir / "umap.png")

        for member in members:
            for ml_model in ml_models:
                member_name = f"{name}-{member}-{ml_model}"
                data_dir = red_dir / member_name
                final_path = data_dir / "final.npy"
                final_data: numpy.ndarray = numpy.load(final_path)
                logger.info(
                    f"Final Data {name}, {member}, {ml_model} shape: {final_data.shape}, "
                    f"dtype: {final_data.dtype}",
                )
                vis.plot_3d.scatter_plot(
                    final_data,
                    labels,
                    data_out_dir / f"{member_name}.png",
                )

                logs_path = data_dir / "logs.npy"
                logs_dir = data_out_dir / "logs"
                logs_dir.mkdir(exist_ok=True)
                vis.plot_2d.plot_logs(logs_path, logs_dir / f"{member_name}.png")


if __name__ == "__main__":
    app()
