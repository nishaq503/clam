"""CLI for the package."""

import logging
import pathlib

import numpy
import typer

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

    datasets = ["cardio"]
    members = ["cc", "gn", "pc", "sc", "vd"]
    ml_models = ["lr", "dt"]
    for name in datasets:
        # The files should be named as `name.npy` and `name_labels.npy`
        data_path = inp_dir / f"{name}.npy"
        labels_path = inp_dir / f"{name}_labels.npy"
        labels: numpy.ndarray = numpy.load(labels_path)
        logger.info(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")

        # Create the dim-reduction for the data and save a scatter plot
        reduced_data = vis.plot_umap.umap_reduce(data_path)
        vis.plot_3d.scatter_plot(reduced_data, labels, out_dir / f"{name}_umap.png")

        # Create a directory for the logs
        logs_dir = out_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        for member in members:
            for ml_model in ml_models:
                # The files should be named as `name_{member}_{ml_model}.npy`
                data_path = red_dir / f"{name}_{member}_{ml_model}.npy"
                data: numpy.ndarray = numpy.load(data_path)
                logger.info(
                    f"Data {name}, {member}, {ml_model} shape: {data.shape}, "
                    f"dtype: {data.dtype}",
                )

                vis.plot_3d.scatter_plot(data, labels, out_dir / f"{name}_{member}_{ml_model}.png")

                logs_path = red_dir / f"{name}_{member}_{ml_model}_logs.npy"
                logs_out_path = logs_dir / f"{name}_{member}_{ml_model}.png"
                vis.plot_2d.plot_logs(logs_path, logs_out_path)


if __name__ == "__main__":
    app()
