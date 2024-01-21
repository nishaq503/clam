"""Search benchmarks for rival algorithms."""

import logging
import pathlib
import numpy

import typer

from . import algorithms
from . import reports

logger = logging.getLogger("rivals")
logger.setLevel("INFO")

app = typer.Typer()


@app.command()
def run_rival(
    rival: algorithms.Rival = typer.Option(
        ...,
        "--rival",
        "-r",
        help="The rival algorithm to run.",
    ),
    data_dir: pathlib.Path = typer.Option(
        ...,
        "--data-dir",
        "-d",
        help="The directory with the data.",
        exists=True,
        readable=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    dataset_name: str = typer.Option(
        ...,
        "--dataset-name",
        "-n",
        help="The name of the dataset.",
    ),
    metric: algorithms.Metric = typer.Option(
        ...,
        "--metric",
        "-m",
        help="The metric to use.",
    ),
    max_search_time: float = typer.Option(
        10.0,
        "--max-search-time",
        "-t",
        help="The maximum time to spend on each query.",
    ),
    cakes_dir: pathlib.Path = typer.Option(
        ...,
        "--cakes-dir",
        "-c",
        help="The directory with the rnn reports from CAKES.",
        exists=True,
        readable=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    output_dir: pathlib.Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="The directory where the reports will be saved.",
        exists=True,
        readable=True,
        writable=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
) -> None:
    """Run a rival algorithm on a dataset and all augmented versions of the dataset."""
    # Get the path of the dataset
    train_path = data_dir / f"{dataset_name}-train.npy"
    test_path = data_dir / f"{dataset_name}-test.npy"

    # Check if the dataset exists
    if not train_path.exists():
        msg = f"Dataset {dataset_name} does not exist in {data_dir}."
        logger.error(msg)
        raise FileNotFoundError(msg)

    # Check if the test set exists
    if not test_path.exists():
        msg = f"Test set for {dataset_name} does not exist in {data_dir}."
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info(f"Found {dataset_name}. Reading scaled versions.")

    # Read all scaled paths.
    # The pattern is <dataset_name>-scale-<scale_factor>-train.npy
    scaled_paths = list(
        (int(path.stem.split("-")[-2]), path)
        for path in data_dir.glob(f"{dataset_name}-scale-*-train.npy")
        if path.exists()
    )
    scaled_paths.sort(key=lambda x: x[0])

    logger.info(f"Found {[s for s, _ in scaled_paths]} scaled versions.")

    ks = [10, 100]

    # load the rnn-reports from CAKES to get radii
    cakes_rnn_reports = reports.CakesRnnReport.from_dir(cakes_dir)
    cakes_rnn_reports = [
        report
        for report in cakes_rnn_reports
        if report.dataset == dataset_name and report.metric == metric.value
    ]

    # prepend the original dataset path to the list of scaled paths
    scaled_paths.insert(0, (1, train_path))

    # Read the test set
    test: numpy.ndarray = numpy.load(test_path)
    num_queries = test.shape[0]
    target_recall = 1.0 - 1e-2

    # Run the algorithm on each dataset
    model = rival.model()()
    for scale, path in scaled_paths:
        # Get the radii for the current scale
        radii = list(
            sorted(
                report.radius for report in cakes_rnn_reports if report.scale == scale
            )
        )
        logger.info(f"Found {radii = } for {scale = }.")

        radii = []
        logger.info("Skipping ranged search ...")

        logger.info(f"Running {rival.value} on {dataset_name} at {scale = }.")

        # Read the dataset
        train: numpy.ndarray = numpy.load(path)

        # Tune the index
        if model.should_tune:
            logger.info(f"Tuning index for {target_recall = }.")
            tuning_time = model.tune_index(
                train,
                metric,
                target_recall,
                test,
                ks,
                radii,
                logger,
                output_dir,
                dataset_name,
                scale,
                max_search_time,
            )
            logger.info(f"Tuning time: {tuning_time:.3f} seconds.")
        else:
            logger.info("Not tuning index.")
            tuning_time = 0.0

        logger.info("Building index.")
        cardinality, dimensionality, index_build_time = model.build_index(train, metric)
        logger.info(f"{index_build_time = :.3f} seconds.")

        # Run knn-search
        for k in ks:
            logger.info(f"Running knn-search with k={k}.")
            throughput, results = model.knn_search(test, k, max_search_time)

            if rival == algorithms.Rival.FaissFlat:
                assert isinstance(model, algorithms.faiss_flat.FaissFlat)
                algorithms.utils.save_knn_results(
                    output_dir, dataset_name, metric, scale, k, results
                )
                recall = 1.0
                logger.info(f"KNN Recall is {recall:.3f} with {throughput:.3f} QPS.")
            else:
                # otherwise, load the ground truth
                true_results = algorithms.utils.load_knn_results(
                    output_dir, dataset_name, metric, scale, k
                )

                recall = algorithms.utils.compute_recall(true_results, results)
                logger.info(f"KNN Recall is {recall:.3f} with {throughput:.3f} QPS.")

            # Create and save the knn-report
            logger.info(f"Saving {rival.value} knn-report.")
            reports.KnnReport(
                algorithm=rival.value,
                dataset=dataset_name,
                scale=scale,
                metric=metric.value,
                cardinality=cardinality,
                dimensionality=dimensionality,
                tuning_time=tuning_time,
                tuned_params=model.tuned_params,
                index_build_time=index_build_time,
                num_queries=num_queries,
                k=k,
                throughput=throughput,
                recall=recall,
            ).save(output_dir)

        if model.has_rnn:
            # Run rnn-search
            for radius in radii:
                logger.info(f"Running rnn-search with {radius = }.")
                throughput, results = model.rnn_search(test, radius, max_search_time)

                if rival == algorithms.Rival.FaissFlat:
                    assert isinstance(model, algorithms.faiss_flat.FaissFlat)
                    algorithms.utils.save_rnn_results(
                        output_dir, dataset_name, metric, scale, radius, results
                    )
                    recall = 1.0
                    logger.info(
                        f"RNN Recall is {recall:.3f} with {throughput:.3f} QPS."
                    )
                else:
                    # otherwise, load the ground truth
                    true_results = algorithms.utils.load_rnn_results(
                        output_dir, dataset_name, metric, scale, radius
                    )

                    recall = algorithms.utils.compute_recall(true_results, results)
                    logger.info(
                        f"RNN Recall is {recall:.3f} with {throughput:.3f} QPS."
                    )

                # Create and save the rnn-report
                logger.info(f"Saving {rival.value} rnn-report.")
                reports.RnnReport(
                    algorithm=rival.value,
                    dataset=dataset_name,
                    scale=scale,
                    metric=metric.value,
                    cardinality=cardinality,
                    dimensionality=dimensionality,
                    tuning_time=tuning_time,
                    tuned_params=model.tuned_params,
                    index_build_time=index_build_time,
                    num_queries=num_queries,
                    radius=radius,
                    throughput=throughput,
                    recall=recall,
                ).save(output_dir)
