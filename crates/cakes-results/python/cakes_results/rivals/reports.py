"""Parser for the scaling results of the Cakes search."""

import json
import pathlib
import typing

import pydantic


class CakesKnnReport(pydantic.BaseModel):
    """Report of knn-search by CAKES."""

    dataset: str
    scale: typing.Optional[int]
    metric: str
    cardinality: int
    dimensionality: int
    shard_sizes: list[int]
    num_queries: int
    tree_build_time: float
    tuning_parameters: typing.Optional[tuple[int, int]]
    tuning_time: typing.Optional[float]
    k: int
    algorithm_throughput: list[tuple[str, float]]
    linear_throughput: float

    @staticmethod
    def from_path(path: pathlib.Path) -> "CakesKnnReport":
        """Parse knn report from a file."""
        with path.open() as f:
            contents: dict = json.load(f)
            if "scale" not in contents:
                contents["scale"] = None
            report = CakesKnnReport(**contents)
        dataset = report.dataset

        if "-scale-" in dataset:
            parts = dataset.split("-scale-")
            report.scale = int(parts[1])
            report.dataset = parts[0]

        if "random" in dataset:
            cardinality = int(dataset.split("-")[1])
            report.scale = cardinality // 1_000_000
            report.dataset = "random-1000000-128"

        return report

    @staticmethod
    def from_dir(path: pathlib.Path) -> list["CakesKnnReport"]:
        """Parse knn reports from a directory."""
        return [
            CakesKnnReport.from_path(p)
            for p in path.iterdir()
            if p.name.startswith("knn-") and p.name.endswith(".json")
        ]


class CakesRnnReport(pydantic.BaseModel):
    """Report of rnn-search by CAKES."""

    dataset: str
    scale: typing.Optional[int]
    metric: str
    cardinality: int
    dimensionality: int
    shard_sizes: list[int]
    num_queries: int
    radius: float
    throughput: float
    linear_throughput: float

    @staticmethod
    def from_path(path: pathlib.Path) -> "CakesRnnReport":
        """Parse rnn report from a file."""
        with path.open() as f:
            contents = json.load(f)
            if "scale" not in contents:
                contents["scale"] = None
            report = CakesRnnReport(**contents)
        dataset = report.dataset

        if "-scale-" in dataset:
            parts = dataset.split("-scale-")
            report.scale = int(parts[1])
            report.dataset = parts[0]

        if "random" in dataset:
            cardinality = int(dataset.split("-")[1])
            report.scale = cardinality // 1_000_000
            report.dataset = "random-1000000-128"

        return report

    @staticmethod
    def from_dir(path: pathlib.Path) -> list["CakesRnnReport"]:
        """Parse rnn reports from a directory."""
        return [
            CakesRnnReport.from_path(p)
            for p in path.iterdir()
            if p.name.startswith("rnn-") and p.name.endswith(".json")
        ]


class KnnReport(pydantic.BaseModel):
    """Report of knn-search on augmented data."""

    algorithm: str
    dataset: str
    scale: int
    metric: str
    cardinality: int
    dimensionality: int
    tuning_time: float
    tuned_params: dict[str, typing.Any]
    index_build_time: float
    num_queries: int
    k: int
    throughput: float
    recall: float

    @staticmethod
    def from_dir(path: pathlib.Path) -> list["KnnReport"]:
        """Parse knn reports from a directory."""
        reports = []
        for p in path.iterdir():
            if p.name.startswith("results-knn-") and p.name.endswith(".json"):
                with p.open() as f:
                    contents = json.load(f)
                    if "scale" not in contents:
                        contents["scale"] = None
                    reports.append(KnnReport(**contents))
        return reports

    def save(self, path: pathlib.Path) -> None:
        """Save report to a file."""
        name = (
            f"results-knn-{self.algorithm}-{self.dataset}-{self.scale}-{self.metric}"
            f"-{self.k}.json"
        )
        with (path / name).open("w") as f:
            json.dump(self.model_dump(), f, indent=2)


class RnnReport(pydantic.BaseModel):
    """Report of rnn-search on augmented data."""

    algorithm: str
    dataset: str
    scale: int
    metric: str
    cardinality: int
    dimensionality: int
    tuning_time: float
    tuned_params: dict[str, typing.Any]
    index_build_time: float
    num_queries: int
    radius: float
    throughput: float
    recall: float

    @staticmethod
    def from_dir(path: pathlib.Path) -> list["RnnReport"]:
        """Parse rnn reports from a directory."""
        reports = []
        for p in path.iterdir():
            if p.name.startswith("results-rnn-") and p.name.endswith(".json"):
                with p.open() as f:
                    reports.append(RnnReport(**json.load(f)))
        return reports

    def save(self, path: pathlib.Path) -> None:
        """Save report to a file."""
        name = (
            f"results-rnn-{self.algorithm}-{self.dataset}-{self.scale}-{self.metric}"
            f"-{self.radius}.json"
        )
        with (path / name).open("w") as f:
            json.dump(self.model_dump(), f, indent=2)
