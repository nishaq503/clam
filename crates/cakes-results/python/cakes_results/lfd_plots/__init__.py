"""Create LFD plots for the Cakes trees."""

import logging
import pathlib
from matplotlib import pyplot as plt
import numpy

import pandas
import typer

# Initialize the logger
logger = logging.getLogger("lfd-plots")
logger.setLevel("INFO")

app = typer.Typer()


@app.command()
def create_plots(
    input_dir: pathlib.Path = typer.Option(
        ...,
        "--input-dir",
        "-i",
        help="The directory with the clusters' csv files.",
        exists=True,
        readable=True,
        file_okay=False,
        resolve_path=True,
    ),
    add_title: bool = typer.Option(
        False,
        "--add-title",
        "-t",
        help="Add a title to the plots.",
    ),
    output_dir: pathlib.Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="The directory to save the plots.",
        exists=True,
        writable=True,
        file_okay=False,
        resolve_path=True,
    ),
) -> None:
    """Create the plots for the LFD results of the Cakes search."""

    logger.info(f"input_dir = {input_dir}")
    logger.info(f"add_title = {add_title}")
    logger.info(f"output_dir = {output_dir}")

    # Read all the csv files in the input directory
    csv_paths = list(input_dir.glob("*.csv"))

    # For radio-ml and silva, remove the csv files with cardinality smaller than
    # the maximum
    max_radio_ml = max(
        [int(path.name.split("-")[1]) for path in csv_paths if "radio" in path.name]
    )
    radio_path = [
        path
        for path in csv_paths
        if "radio" in path.name and f"{max_radio_ml}" in path.name
    ][0]

    max_silva = max(
        [int(path.name.split("-")[1]) for path in csv_paths if "silva" in path.name]
    )
    silva_path = [
        path
        for path in csv_paths
        if "silva" in path.name and f"{max_silva}" in path.name
    ][0]

    ann_paths = [
        path
        for path in csv_paths
        if "radio" not in path.name and "silva" not in path.name
    ]
    csv_paths = [radio_path, silva_path, *ann_paths]

    # CSV columns:
    # "id",
    # "depth",
    # "offset",
    # "cardinality",
    # "is_leaf",
    # "radius",
    # "lfd",
    # "polar_distance",
    # "ratio_cardinality",
    # "ratio_radius",
    # "ratio_lfd",
    # "ratio_cardinality_ema",
    # "ratio_radius_ema",
    # "ratio_lfd_ema",
    properties = [
        "cardinality",
        "radius",
        "lfd",
        # "polar_distance",
        # "ratio_cardinality",
        # "ratio_radius",
        # "ratio_lfd",
        # "ratio_cardinality_ema",
        # "ratio_radius_ema",
        # "ratio_lfd_ema",
        "fractal_density",
        "log_car_by_lfd",
    ]

    for path in csv_paths:
        if "silva" in path.name:
            dataset = "silva"
            cardinality = int(path.name.split("-")[1])
        elif "radio" in path.name:
            dataset = "radio-ml"
            cardinality = int(path.name.split("-")[1])
        else:
            parts = path.stem.split("-")
            cardinality = int(parts[-2])
            dataset = "-".join(parts[:-2])

        if "random" in dataset:
            dataset = "random"

        # For each property, create a plot
        for property in properties:
            plot_deciles(
                dataset=dataset,
                cardinality=cardinality,
                prop=property,
                csv_path=path,
                add_title=add_title,
                output_dir=output_dir,
            )


def plot_deciles(
    *,
    dataset: str,
    cardinality: int,
    prop: str,
    csv_path: pathlib.Path,
    add_title: bool,
    output_dir: pathlib.Path,
) -> None:
    """Plot the deciles of the given property."""

    logger.info(f"Plotting {dataset = }, {cardinality = }, {prop = } ...")

    # Read the csv file
    inp_df = pandas.read_csv(csv_path)

    # # Remove all rows with leaf clusters
    # inp_df = inp_df[~inp_df["is_leaf"]]

    # Add a column for the normalized radius
    max_radius = inp_df["radius"].max()
    inp_df["radius"] = (inp_df["radius"] / max_radius) + 1e-8

    if prop == "fractal_density":
        inp_df["fractal_density"] = inp_df["cardinality"].apply(
            lambda x: numpy.log(x)
        ) - inp_df["lfd"] * inp_df["radius"].apply(lambda x: numpy.log(x))
        inp_df["fractal_density"] = inp_df["fractal_density"].apply(
            lambda x: numpy.exp(x)
        )

    elif prop == "log_car_by_lfd":
        inp_df["log_car_by_lfd"] = (
            inp_df["cardinality"].apply(lambda x: numpy.log(x)) / inp_df["lfd"]
        )

    # drop all columns except depth, cardinality and the property
    if prop == "cardinality":
        inp_df = inp_df[["depth", "cardinality"]]
    else:
        inp_df = inp_df[["depth", "cardinality", prop]]

    # Create an empty dataframe to store the deciles, min and max
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    quantile_names = [f"quantile-{int(q * 100)}" for q in quantiles]
    column_names = ["depth", "min", *quantile_names, "max"]
    out_df = pandas.DataFrame(columns=column_names)

    for depth, group in inp_df.groupby("depth"):
        # sort the group by the property
        group.sort_values(prop, inplace=True)

        # get the minimum and maximum values
        prop_min = group[prop].min()
        prop_max = group[prop].max()

        if prop in ["radius", "lfd", "fractal_density", "log_car_by_lfd"]:
            # Get the cumulative sum of the cardinality
            order = group["cardinality"].cumsum()
            order = order / order.max()

            deciles = []
            for q in quantiles:
                index = order[order >= q].index[0]
                v = group.loc[index, prop]
                deciles.append(v)
        else:
            deciles = group[prop].quantile(quantiles)

        out_df.loc[depth] = [depth, prop_min, *deciles, prop_max]

    # Plot the deciles
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(10, 5))

    for name in reversed(column_names[1:]):
        if name in ["min", "max"]:
            label = f"{name}imum value"
        else:
            q = int(name.split("-")[1])
            label = f"{q:02d}th percentile"

        ax.plot(
            out_df["depth"],
            out_df[name],
            label=label,
            # make the line dashed if it is not the median
            linestyle="dashed" if "quantile" not in name else "solid",
            # make the line thicker if it is the median
            linewidth=1 if "quantile-50" in name else 0.5,
        )

    # fill the area between the min and max
    ax.fill_between(out_df["depth"], out_df["min"], out_df["max"], alpha=0.1)

    # fill the area between pairs of quantiles with increasing opacity
    for i, q in enumerate(quantiles[: len(quantiles) // 2], start=1):
        q_min = int(q * 100)
        q_max = 100 - q_min
        ax.fill_between(
            out_df["depth"],
            out_df[f"quantile-{q_min}"],
            out_df[f"quantile-{q_max}"],
            alpha=i / 5,
        )

    if prop == "fractal_density":
        prop_label = "cardinality / (normalized_radius ^ lfd)"
    elif prop == "log_car_by_lfd":
        prop_label = "log(cardinality) / lfd"
    else:
        prop_label = prop

    # Set the labels for both axes
    ax.set_xlabel("Depth")
    ax.set_ylabel(f"{prop_label}")

    # deal with the y-axis
    if prop in ["polar_distance", "fractal_density", "cardinality"]:
        # set the y-axis to a log scale
        ax.set_yscale("log")
    else:
        if prop == "lfd":
            max_value = 21
            ax.set_ylim(top=max_value)
            ax.set_yticks(range(0, max_value, 2))

        # add horizontal grid lines
        ax.grid(axis="y")

    if add_title:
        ax.set_title(f"{dataset} - {cardinality} - {prop_label}")

    # Add a legend in the upper right corner
    ax.legend(loc="upper right")

    prop_dir = output_dir.joinpath(prop)
    prop_dir.mkdir(exist_ok=True)

    # Save the figure
    fig.tight_layout()
    fig.savefig(prop_dir.joinpath(f"{dataset}-{cardinality}.png"), dpi=300)
    plt.close(fig)
