"""Helpers for dealing with the datasets used in the paper."""

import enum
import pathlib

import numpy


class Metric(enum.StrEnum):
    """Enum of distance metrics used in the paper."""
    Euclidean = "euclidean"
    Cosine = "cosine"

    def faiss_name(self) -> str:
        """Get the FAISS name for the metric."""
        if self == Metric.Euclidean:
            return "L2"
        if self == Metric.Cosine:
            return "IP"
        raise ValueError(f"Unknown metric: {self.value}")


class Dataset(enum.StrEnum):
    """Enum of datasets used in the paper."""
    # Euclidean
    FashionMnist = "fashion-mnist"
    Mnist = "mnist"
    Sift = "sift"
    Gist = "gist"
    # Cosine
    Glove25 = "glove-25"
    Glove50 = "glove-50"
    Glove100 = "glove-100"
    Glove200 = "glove-200"
    DeepImage = "deep-image"
    NYTimes = "nytimes"
    LastFM = "lastfm"

    def metric(self) -> Metric:
        """Get the distance metric associated with the dataset."""
        if self in {
            Dataset.FashionMnist,
            Dataset.Mnist,
            Dataset.Sift,
            Dataset.Gist,
        }:
            return Metric.Euclidean
        if self in {
            Dataset.Glove25,
            Dataset.Glove50,
            Dataset.Glove100,
            Dataset.Glove200,
            Dataset.DeepImage,
            Dataset.NYTimes,
            Dataset.LastFM,
        }:
            return Metric.Cosine
        raise ValueError(f"Unknown dataset: {self.value}")

    @staticmethod
    def from_name(name: str) -> "Dataset":
        """Get the Dataset enum member from its name."""
        name = name.lower()
        for dataset in Dataset:
            if dataset.value == name:
                return dataset
        raise ValueError(f"Unknown dataset name: {name}")

    def __subset_path(self, base_dir: pathlib.Path, subset: str) -> pathlib.Path:
        """Get the path to the specified subset (train/test) of the dataset."""
        name = f"{self.value}-{subset}.npy"
        path = base_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return path

    def __read_subset(self, base_dir: pathlib.Path, subset: str, rng: numpy.random.Generator | None = None) -> numpy.ndarray:
        """Read the specified subset (train/test) of the dataset."""
        path = self.__subset_path(base_dir, subset)
        data = numpy.load(path)
        if rng is not None:
            rng.shuffle(data)
        return data

    def read_train(self, base_dir: pathlib.Path, rng: numpy.random.Generator | None = None) -> numpy.ndarray:
        """Read the training subset of the dataset.

        Arguments:
            base_dir: The base directory where the dataset files are located.
            rng: A random number generator for shuffling the data. If None, no shuffling is done.

        Returns:
            The training data as a NumPy array.
        """
        return self.__read_subset(base_dir, "train", rng)

    def read_test(self, base_dir: pathlib.Path, rng: numpy.random.Generator | None = None) -> numpy.ndarray:
        """Read the test subset of the dataset.

        Arguments:
            base_dir: The base directory where the dataset files are located.
            rng: A random number generator for shuffling the data. If None, no shuffling is done.

        Returns:
            The test data as a NumPy array.
        """
        return self.__read_subset(base_dir, "test", rng)
